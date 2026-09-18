#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
outlet_flow_analysis_3.py
=========================

Script per l'analisi dei flussi di uscita (CFD post-EVAR) e confronto con la letteratura.
Integra le seguenti 4 MODIFICHE METODOLOGICHE CRITICHE:

  1. ALLINEAMENTO TEMPORALE ANCORATO ALL'INLET (Peak Alignment Vincolato):
     Non trasla i rami indipendentemente (cosa che cancellerebbe il tempo di transito
     fisiologico dell'onda sfigmica), ma applica LO STESSO SHIFT TEMPORALE calcolato
     sul picco dell'inlet (Sopraceliaco / SC) a tutti i rami di ciascun paziente.

  2. NORMALIZZAZIONE FISIOLOGICA PER FLUSSO MEDIO (Q / Q_mean):
     Supporta la normalizzazione 'mean' (default o configurabile) che preserva
     la componente continua diastolica (offset) senza snaturare la differenza
     tra rami ad alta resistenza (iliache) e bassa resistenza (renali/viscerali).

  3. TEST TOST CON MARGINE DINAMICO FISIOLOGICO:
     Il margine di equivalenza viene calcolato in modo difendibile basandosi
     sulla variabilità fisiologica di Les et al. (0.5 * SD_Les ~ 0.042, ossia 4.2%).

  4. DIAGNOSTICA AVANZATA DEL REVERSE FLOW:
     Distingue chiaramente l'inversione di flusso fisiologica (decelerazione
     sistolica/onda sfigmica retrograda nell'aorta e iliache) dagli artefatti
     numerici/RCR viscerali (es. reflusso patologico sui rami renali).
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats


# ============================================================================
# RIFERIMENTI DI LETTERATURA
# ============================================================================

# Les et al., Cardiovasc Eng Technol 2010 (model cohort n=36)
LES_N = 36
LES_IRSC_RATIO_MEAN = 0.343
LES_IRSC_RATIO_SD = 0.0843

# Margine TOST fisiologico derivato direttamente dalla deviazione standard di Les et al.
DEFAULT_TOST_MARGIN = 0.5 * LES_IRSC_RATIO_SD  # ~ 0.04215 (4.2%)

LES_SC_WAVE = np.array([11.8, 13.1, 19.5, 96.1, 204., 203., 172., 136., 103., 60.9,
                        20.2, 4.75, 9.19, 15.0, 18.6, 18.9, 18.4, 18.0, 15.3, 14.7,
                        13.7, 14.0, 13.7, 14.6])
LES_IR_WAVE = np.array([-1.08, -1.80, 2.88, 32.4, 99.6, 114., 92.8, 66.7, 41.1, 15.4,
                        -7.55, -14.2, -10.9, -5.71, -1.67, 0.333, 0.702, 0.594, -0.497,
                        -0.481, -1.17, -0.306, -0.951, 0.418])

# Aggregato viscere+reni = SC - IR (conservazione di massa frame-by-frame).
LES_VR_WAVE = LES_SC_WAVE - LES_IR_WAVE

# Riferimenti di FORMA
REF_WAVES = {"inlet": LES_SC_WAVE, "visceral_renal": LES_VR_WAVE, "iliac_total": LES_IR_WAVE}
REF_LABEL = {"inlet": "Les SC", "visceral_renal": "Les SC-IR", "iliac_total": "Les IR"}
VR_MEMBERS = ["celiac", "sma", "renal_L", "renal_R"]

# Tesi Surianarayanan 2020 (1 paziente, SENZA SD)
THESIS_FRACTION = {"celiac": 0.218, "sma": 0.147, "renal_L": 0.147, "renal_R": 0.147}


# ============================================================================
# CONFIGURAZIONE
# ============================================================================

@dataclass
class Config:
    territories: dict = field(default_factory=lambda: {
        "celiac":  ["celiac"],
        "sma":     ["sma"],
        "renal_L": ["renal_L"],
        "renal_R": ["renal_R"],
        "iliac_total": ["iliac_L", "iliac_R",
                        "iliac_L_ext", "iliac_L_int",
                        "iliac_R_ext", "iliac_R_int"],
    })
    inlet_label: str = "inlet"
    n_resample: int = 100
    cardiac_period: float | None = 0.8
    # MODIFICA 3: Usiamo il margine dinamico fisiologico derivato da Les SD anziché 0.05 fisso
    tost_margin: float = DEFAULT_TOST_MARGIN
    alpha: float = 0.05
    reverse_flag_threshold: float = 0.02
    # MODIFICA 2: Di default usiamo 'mean' (divide per la media) per non perdere la componente continua diastolica
    shape_norm: str = "mean"
    shape_align: bool = True


# ============================================================================
# EXPORT MODE
# ============================================================================

def export_mode(args):
    """Estrae dai VTU e scrive CSV per paziente + template etichette."""
    import io_vtu
    from config import Config
    cfg = Config()
    os.makedirs(args.export_dir, exist_ok=True)
    patient_dirs = io_vtu.list_patients(cfg)
    if not patient_dirs:
        print("Nessun paziente trovato con la config attuale."); sys.exit(1)

    names_info = {}
    for pdir in patient_dirs:
        patient = os.path.basename(os.path.normpath(pdir))
        print(f"[export] {patient}")
        faces = io_vtu.read_faces(pdir, cfg)
        times, P, Q, match_d = io_vtu.extract_timeseries(pdir, faces, cfg)

        cols = {"time": times}
        for f in faces.values():
            if f.kind not in ("inlet", "outlet"):
                continue
            cols[f.name] = Q[f.name]
            info = names_info.setdefault(f.name, {"kind": f.kind, "radii": [], "count": 0})
            info["radii"].append(f.radius)
            info["count"] += 1
            c = f.points.mean(axis=0)
            info["centroid"] = (float(c[0]), float(c[1]), float(c[2]))
        pd.DataFrame(cols).to_csv(os.path.join(args.export_dir, f"{patient}.csv"), index=False)

    rows = []
    for name, info in sorted(names_info.items()):
        rows.append({
            "raw_outlet": name,
            "kind": info["kind"],
            "n_patients": info["count"],
            "median_radius_cm": float(np.median(info["radii"])),
            "cx": info["centroid"][0], "cy": info["centroid"][1], "cz": info["centroid"][2],
            "label": "inlet" if info["kind"] == "inlet" else "",
            "group": "inlet" if info["kind"] == "inlet" else "",
        })
    tpl = os.path.join(args.export_dir, "label_map_TEMPLATE.csv")
    pd.DataFrame(rows).to_csv(tpl, index=False)
    print(f"\n[export] CSV e Template generati con successo in {args.export_dir}")


# ============================================================================
# CARICAMENTO DATI
# ============================================================================

def load_label_map(path):
    df = pd.read_csv(path)
    if not {"raw_outlet", "label"}.issubset(df.columns):
        raise ValueError("label_map deve avere almeno colonne raw_outlet,label")
    df = df[df["label"].astype(str).str.strip() != ""]
    generic = {r["raw_outlet"]: r["label"] for _, r in df.iterrows()
               if "patient" not in df.columns or pd.isna(r.get("patient"))}
    specific = {}
    if "patient" in df.columns:
        for _, r in df.iterrows():
            if not pd.isna(r.get("patient")):
                specific[(str(r["patient"]), r["raw_outlet"])] = r["label"]
    return generic, specific


def resolve_label(patient, raw, generic, specific):
    if (patient, raw) in specific:
        return specific[(patient, raw)]
    return generic.get(raw)


def load_patient_csv(patient, data_root, generic, specific):
    df = pd.read_csv(os.path.join(data_root, f"{patient}.csv"))
    tcol = "time" if "time" in df.columns else df.columns[0]
    t = df[tcol].to_numpy(float)
    curves = {}
    for col in df.columns:
        if col == tcol:
            continue
        lab = resolve_label(patient, col, generic, specific)
        if lab is None:
            continue
        Q = df[col].to_numpy(float)
        if lab in curves:
            curves[lab] = (t, curves[lab][1] + Q)
        else:
            curves[lab] = (t, Q)
    return curves


# ============================================================================
# ELABORAZIONE FISIOLOGICA E METRICHE
# ============================================================================

def extract_last_cycle(t, Q, period):
    if period is None or period >= (t[-1] - t[0]):
        return t - t[0], Q
    mask = t >= (t[-1] - period - 1e-12)
    tt, QQ = t[mask], Q[mask]
    return tt - tt[0], QQ


def cycle_mean(t, Q):
    T = t[-1] - t[0]
    return float(np.mean(Q)) if T <= 0 else float(np.trapezoid(Q, t) / T)


def orient_positive(t, Q):
    return (-Q, True) if cycle_mean(t, Q) < 0 else (Q, False)


def resample_cycle(t, Q, n):
    tn = (t - t[0]) / (t[-1] - t[0])
    grid = np.linspace(0.0, 1.0, n)
    return grid, np.interp(grid, tn, Q)


def shape_features(t, Q, mean_flow, label_name=""):
    """
    MODIFICA 4: Diagnostica Avanzata Reverse Flow.
    Calcola le metriche di forma e categorizza l'inversione di flusso
    distinguendo la fisiologia dall'artefatto numerico RCR.
    """
    T = t[-1] - t[0]
    mf = abs(mean_flow) if abs(mean_flow) > 1e-12 else 1e-12
    pos, neg = np.clip(Q, 0, None), np.clip(Q, None, 0)
    fwd = float(np.trapezoid(pos, t)); rev = float(-np.trapezoid(neg, t))
    tn = (t - t[0]) / T if T > 0 else t
    dia = Q[tn >= 0.5]
    
    rev_time_frac = float(np.mean(Q < 0))
    rev_vol_frac = (rev / fwd if fwd > 1e-12 else np.nan)
    min_val = float(np.min(Q))
    
    # Classificazione fisiologica vs artefatto
    if min_val >= 0:
        rev_class = "Assente (flusso sempre anterogrado)"
    elif label_name in ["iliac_total", "iliac_L", "iliac_R", "inlet"] and rev_time_frac < 0.30:
        rev_class = "Inversione FISIOLOGICA (decelerazione sistolica / onda sfigmica)"
    elif label_name in ["renal_L", "renal_R", "celiac", "sma", "visceral_renal"]:
        rev_class = "POSSIBILE ARTEFATTO RCR (Reflusso patologico su distretto viscerale/renale)"
    else:
        rev_class = "Anomalia Severa / Artefatto Numerico"

    return dict(
        pulsatility=(float(np.max(Q)) - float(np.min(Q))) / mf,
        rev_vol_frac=rev_vol_frac,
        rev_time_frac=rev_time_frac,
        t_peak=(float((t[np.argmax(Q)] - t[0]) / T) if T > 0 else np.nan),
        dia_level=(float(np.mean(dia) / mf) if dia.size else np.nan),
        reverse_classification=rev_class
    )


def process_patient(patient, curves, cfg):
    """
    MODIFICA 1: Allineamento temporale vincolato all'inlet.
    Calcola lo shift di allineamento sul solo inlet e lo applica a tutte le uscite.
    """
    if cfg.inlet_label not in curves:
        raise ValueError(f"{patient}: manca l'inlet '{cfg.inlet_label}'")
        
    ti, Qi = extract_last_cycle(*curves[cfg.inlet_label], cfg.cardiac_period)
    Qi_o, _ = orient_positive(ti, Qi)
    inlet_mean = cycle_mean(ti, Qi_o)

    # 1. Calcola lo shift sul picco dell'INLET
    _, Qin_res = resample_cycle(ti, Qi_o, cfg.n_resample)
    inlet_shift = -np.argmax(Qin_res)

    rec = {"patient": patient, "inlet_mean": inlet_mean}
    fractions, features, norm_curves = {}, {}, {}
    outlet_sum = 0.0

    # 2. Applica lo shift rigido vincolato dell'inlet a TUTTE le uscite
    for label, (t, Q) in curves.items():
        if label == cfg.inlet_label:
            continue
        tt, QQ = extract_last_cycle(t, Q, cfg.cardiac_period)
        QQ_o, _ = orient_positive(tt, QQ)
        m = cycle_mean(tt, QQ_o)
        outlet_sum += abs(m)
        fractions[label] = abs(m) / abs(inlet_mean) if abs(inlet_mean) > 1e-12 else np.nan
        features[label] = shape_features(tt, QQ_o, m, label_name=label)
        
        _, Qres = resample_cycle(tt, QQ_o, cfg.n_resample)
        # Shift vincolato all'inlet per preservare la propagazione d'onda
        if cfg.shape_align:
            Qres = np.roll(Qres, inlet_shift)
        norm_curves[label] = Qres

    rec["mass_closure"] = outlet_sum / abs(inlet_mean) if abs(inlet_mean) > 1e-12 else np.nan
    
    # Salva l'inlet allineato
    if cfg.shape_align:
        Qin_res = np.roll(Qin_res, inlet_shift)
    norm_curves[cfg.inlet_label] = Qin_res

    # Aggregati
    iliac_members = cfg.territories.get("iliac_total", [])
    for agg_label, members in [("iliac_total", iliac_members), ("visceral_renal", VR_MEMBERS)]:
        abs_curves = []
        for label, (t, Q) in curves.items():
            if label in members:
                tt, QQ = extract_last_cycle(t, Q, cfg.cardiac_period)
                QQ_o, _ = orient_positive(tt, QQ)
                _, Qres = resample_cycle(tt, QQ_o, cfg.n_resample)
                if cfg.shape_align:
                    Qres = np.roll(Qres, inlet_shift)
                abs_curves.append(Qres)
        if abs_curves:
            norm_curves[agg_label] = np.sum(np.vstack(abs_curves), axis=0)

    return rec, fractions, features, norm_curves


def aggregate(records, fractions_all, cfg):
    labels = sorted({l for f in fractions_all.values() for l in f})
    rows = [{"patient": p, **{l: fr.get(l, np.nan) for l in labels}}
            for p, fr in fractions_all.items()]
    frac_df = pd.DataFrame(rows).set_index("patient")
    terr = pd.DataFrame(index=frac_df.index)
    for tname, members in cfg.territories.items():
        present = [m for m in members if m in frac_df.columns]
        if present:
            terr[tname] = frac_df[present].sum(axis=1, min_count=1)
    vr = [t for t in ["celiac", "sma", "renal_L", "renal_R"] if t in terr.columns]
    if vr:
        terr["visceral_renal"] = terr[vr].sum(axis=1, min_count=1)
    return frac_df, terr


# ============================================================================
# STATISTICA (TOST, Welch, Coverage)
# ============================================================================

def tost_one_sample(x, target, margin, alpha=0.05):
    """MODIFICA 3: TOST con margine derivato da SD_Les."""
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    n = len(x); mean = float(np.mean(x)); sd = float(np.std(x, ddof=1))
    se = sd / np.sqrt(n); df = n - 1
    p_low = stats.t.sf((mean - (target - margin)) / se, df)
    p_up = stats.t.cdf((mean - (target + margin)) / se, df)
    p = max(p_low, p_up)
    return dict(n=n, mean=mean, sd=sd, p_tost=p, equivalent=(p < alpha),
                bounds=(target - margin, target + margin))


def welch_from_summary(m1, s1, n1, m2, s2, n2):
    se = np.sqrt(s1**2 / n1 + s2**2 / n2)
    t = (m1 - m2) / se
    df = se**4 / ((s1**2 / n1)**2 / (n1 - 1) + (s2**2 / n2)**2 / (n2 - 1))
    p = 2 * stats.t.sf(abs(t), df)
    sp = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    tc = stats.t.ppf(0.975, df)
    return dict(t=t, df=df, p=p, cohens_d=(m1 - m2) / sp, diff=m1 - m2,
                ci95=((m1 - m2) - tc * se, (m1 - m2) + tc * se))


def coverage(x, ref_mean, ref_sd):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    lo, hi = ref_mean - 1.96 * ref_sd, ref_mean + 1.96 * ref_sd
    return float(np.mean((x >= lo) & (x <= hi))), (lo, hi)


def run_statistics(terr, cfg):
    out = {}
    if "iliac_total" in terr.columns:
        x = terr["iliac_total"].dropna().to_numpy()
        cov, band = coverage(x, LES_IRSC_RATIO_MEAN, LES_IRSC_RATIO_SD)
        out["level1_iliac"] = dict(
            coverage=cov, ref_band=band,
            tost=tost_one_sample(x, LES_IRSC_RATIO_MEAN, cfg.tost_margin, cfg.alpha),
            welch=welch_from_summary(float(np.mean(x)), float(np.std(x, ddof=1)), len(x),
                                     LES_IRSC_RATIO_MEAN, LES_IRSC_RATIO_SD, LES_N),
            ref_mean=LES_IRSC_RATIO_MEAN, ref_sd=LES_IRSC_RATIO_SD)
    if "visceral_renal" in terr.columns:
        x = terr["visceral_renal"].dropna().to_numpy()
        ref = 1.0 - LES_IRSC_RATIO_MEAN
        cov, band = coverage(x, ref, LES_IRSC_RATIO_SD)
        out["level1_visceral_renal"] = dict(
            coverage=cov, ref_band=band,
            tost=tost_one_sample(x, ref, cfg.tost_margin, cfg.alpha), ref_mean=ref)
    lvl2 = []
    for label, ref in THESIS_FRACTION.items():
        if label in terr.columns:
            x = terr[label].dropna().to_numpy()
            m, sd = float(np.mean(x)), float(np.std(x, ddof=1))
            lvl2.append(dict(ramo=label, coorte_mean=m, coorte_sd=sd, tesi_ref=ref,
                             scarto_abs=m - ref, scarto_rel=(m - ref) / ref))
    out["level2_thesis"] = pd.DataFrame(lvl2)
    return out


# ============================================================================
# NORMALIZZAZIONE E FORMA
# ============================================================================

def normalize_curve(y, mode):
    """
    MODIFICA 2: Normalizzazione 'mean' vs 'pulse'.
    'mean' -> divide per la media (Q/Q_mean), preserva la componente continua diastolica.
    'pulse' -> demean + divide per picco-picco (forma adimensionale pura).
    """
    if mode == "mean":
        m = float(np.mean(y))
        return y / m if abs(m) > 1e-12 else y
    # pulse
    y0 = y - float(np.mean(y))
    pp = float(y.max() - y.min())
    return y0 / pp if pp > 1e-12 else y0


def _ref_norm(wave, n, mode):
    _, r = resample_cycle(np.linspace(0, 1, len(wave)), wave, n)
    return normalize_curve(r, mode)


def shape_metrics_vs_les(norm_all, cfg):
    n = cfg.n_resample
    rows = []
    for agg, wave in REF_WAVES.items():
        ref = _ref_norm(wave, n, cfg.shape_norm)
        for pat, c in norm_all.items():
            if agg not in c:
                continue
            y = normalize_curve(c[agg], cfg.shape_norm)
            corr = float(np.corrcoef(y, ref)[0, 1]) if np.std(y) > 0 else np.nan
            rmse = float(np.sqrt(np.mean((y - ref) ** 2)))
            rows.append({"aggregato": agg, "patient": pat,
                         "corr_vs_les": corr, "rmse_vs_les": rmse})
    return pd.DataFrame(rows)


# ============================================================================
# VISUALIZZAZIONE
# ============================================================================

def _ylabel(cfg):
    return ("flusso / media di ciclo [-]" if cfg.shape_norm == "mean"
            else "flusso demedato / picco-picco [-]")


def spaghetti_per_outlet(norm_all, cfg, outdir):
    os.makedirs(outdir, exist_ok=True)
    grid = np.linspace(0, 1, cfg.n_resample)
    labels = sorted({l for c in norm_all.values() for l in c if l != cfg.inlet_label})
    for label in labels:
        raw = [c[label] for c in norm_all.values() if label in c]
        if not raw:
            continue
        M = np.vstack([normalize_curve(y, cfg.shape_norm) for y in raw])
        mean_c, sd_c = np.nanmean(M, axis=0), np.nanstd(M, axis=0)
        fig, ax = plt.subplots(figsize=(7, 4.3))
        for row in M:
            ax.plot(grid, row, color="0.72", lw=0.7, alpha=0.55)
        ax.plot(grid, mean_c, color="C0", lw=2.4, label=f"media coorte (n={M.shape[0]})")
        ax.fill_between(grid, mean_c - sd_c, mean_c + sd_c, color="C0", alpha=0.18, label="±1 SD")
        if label in REF_WAVES:
            ref = _ref_norm(REF_WAVES[label], cfg.n_resample, cfg.shape_norm)
            ax.plot(grid, ref, color="C3", lw=2.2, ls="--", label=f"{REF_LABEL[label]} (rif.)")
        ax.axhline(0 if cfg.shape_norm == "pulse" else 1.0, color="k", lw=0.8, ls=":")
        alg = "Inlet-Aligned" if cfg.shape_align else "Non allineate"
        ax.set_title(f"Forma normalizzata ({cfg.shape_norm}) - {label} ({alg})")
        ax.set_xlabel("fase del ciclo cardiaco [-]"); ax.set_ylabel(_ylabel(cfg))
        ax.legend(fontsize=8, loc="upper right"); ax.grid(alpha=0.25)
        fig.tight_layout(); fig.savefig(os.path.join(outdir, f"curve_{label}.png"), dpi=140)
        plt.close(fig)


def aggregate_vs_les(norm_all, cfg, outdir, metrics):
    grid = np.linspace(0, 1, cfg.n_resample)
    for agg, wave in REF_WAVES.items():
        raw = [c[agg] for c in norm_all.values() if agg in c]
        if not raw:
            continue
        M = np.vstack([normalize_curve(y, cfg.shape_norm) for y in raw])
        mean_c, sd_c = np.nanmean(M, axis=0), np.nanstd(M, axis=0)
        ref = _ref_norm(wave, cfg.n_resample, cfg.shape_norm)
        fig, ax = plt.subplots(figsize=(7, 4.3))
        ax.plot(grid, mean_c, color="C0", lw=2.4, label=f"{agg} (media, n={M.shape[0]})")
        ax.fill_between(grid, mean_c - sd_c, mean_c + sd_c, color="C0", alpha=0.18, label="±1 SD")
        ax.plot(grid, ref, color="C3", lw=2.2, ls="--", label=f"{REF_LABEL[agg]} (36 AAA)")
        ax.axhline(0 if cfg.shape_norm == "pulse" else 1.0, color="k", lw=0.8, ls=":")
        sub = metrics[metrics["aggregato"] == agg] if not metrics.empty else metrics
        if not sub.empty:
            ax.text(0.02, 0.97, f"corr={sub['corr_vs_les'].mean():.2f}  "
                                f"RMSE={sub['rmse_vs_les'].mean():.2f}",
                    transform=ax.transAxes, va="top", fontsize=9,
                    bbox=dict(boxstyle="round", fc="white", alpha=0.7))
        alg = "Inlet-Aligned" if cfg.shape_align else "Non allineate"
        ax.set_title(f"Forma {agg} vs {REF_LABEL[agg]} ({cfg.shape_norm}, {alg})")
        ax.set_xlabel("fase del ciclo [-]"); ax.set_ylabel(_ylabel(cfg))
        ax.legend(fontsize=8); ax.grid(alpha=0.25)
        fig.tight_layout(); fig.savefig(os.path.join(outdir, f"forma_{agg}_vs_les.png"), dpi=140)
        plt.close(fig)


def fractions_summary(terr, stats_out, cfg, outdir):
    order = [c for c in ["celiac", "sma", "renal_L", "renal_R", "iliac_total"] if c in terr.columns]
    data = [terr[c].dropna().to_numpy() for c in order]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.violinplot(data, showmeans=True, showextrema=True)
    ax.set_xticks(range(1, len(order) + 1)); ax.set_xticklabels(order, rotation=15)
    ax.set_ylabel("frazione del flusso all'inlet [-]")
    ax.set_title("Distribuzione di flusso agli outlet - coorte")
    for i, c in enumerate(order, start=1):
        if c in THESIS_FRACTION:
            ax.plot(i, THESIS_FRACTION[c], "D", color="C1", ms=7,
                    label="tesi (rif.)" if i == 1 else None)
    if "iliac_total" in order and "level1_iliac" in stats_out:
        i = order.index("iliac_total") + 1
        lo, hi = stats_out["level1_iliac"]["ref_band"]
        ax.fill_between([i - 0.4, i + 0.4], lo, hi, color="C3", alpha=0.2, label="Les IR/SC ±1.96 SD")
        ax.plot([i - 0.4, i + 0.4], [LES_IRSC_RATIO_MEAN] * 2, color="C3", lw=2)
    ax.legend(fontsize=8); ax.grid(alpha=0.25, axis="y")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "frazioni_coorte.png"), dpi=140)
    plt.close(fig)


# ============================================================================
# GENERAZIONE REPORT
# ============================================================================

def write_report(records, terr, stats_out, features_all, cfg, outdir):
    L = []; A = L.append
    A("=" * 70); A("REPORT CONFRONTO DISTRIBUZIONE DI FLUSSO AGLI OUTLET"); A("=" * 70)
    A(f"\nPazienti analizzati: {len(records)}")
    mc = np.array([r["mass_closure"] for r in records], float)
    bad = int(np.sum(np.abs(mc - 1) > 0.02))
    A(f"\n[QC] Chiusura di massa: media {np.nanmean(mc):.4f} "
      f"(min {np.nanmin(mc):.4f}, max {np.nanmax(mc):.4f}); {bad} pz oltre 2%.")
    
    if "level1_iliac" in stats_out:
        s = stats_out["level1_iliac"]; t = s["tost"]; w = s["welch"]
        A("\n" + "-" * 70)
        A(f"LIVELLO 1 (robusto) - iliache L+R vs Les IR/SC = {s['ref_mean']:.3f} ± {s['ref_sd']:.3f}")
        A("-" * 70)
        A(f"  coorte: media {t['mean']:.3f} ± {t['sd']:.3f}")
        A(f"  coverage entro [{s['ref_band'][0]:.3f},{s['ref_band'][1]:.3f}]: {s['coverage']*100:.1f}%")
        A(f"  TOST dinamico (±{cfg.tost_margin:.4f} = 0.5*SD_Les): p={t['p_tost']:.4f} -> "
          f"{'EQUIVALENTE' if t['equivalent'] else 'NON equivalente'}")
        A(f"  Welch: diff={w['diff']:+.3f} (IC95 [{w['ci95'][0]:+.3f},{w['ci95'][1]:+.3f}]), "
          f"p={w['p']:.4f}, d={w['cohens_d']:+.2f}")
          
    A("\n" + "-" * 70)
    A("DIAGNOSTICA AVANZATA REVERSE FLOW (Fisiologia vs Artefatti RCR)")
    A("-" * 70)
    for rlabel in ["renal_L", "renal_R", "iliac_total"]:
        classifications = [f[rlabel]["reverse_classification"] for f in features_all.values() if rlabel in f]
        if classifications:
            counts = pd.Series(classifications).value_counts()
            A(f"  [{rlabel}]")
            for cat, count in counts.items():
                A(f"    - {cat}: {count}/{len(classifications)} pazienti")

    rep = "\n".join(L)
    with open(os.path.join(outdir, "report.txt"), "w") as fh:
        fh.write(rep + "\n")
    print(rep)


# ============================================================================
# ANALYZE MODE
# ============================================================================

def analyze_mode(args):
    # Fissa il margine TOST dinamicamente a 0.5 * SD_Les se non sovrascritto da CLI
    tost_m = args.tost_margin if args.tost_margin != 0.05 else DEFAULT_TOST_MARGIN
    
    cfg = Config(
        cardiac_period=args.period,
        tost_margin=tost_m,
        shape_norm=args.shape_norm,
        shape_align=not args.no_align
    )
    os.makedirs(args.outdir, exist_ok=True)
    fig_dir = os.path.join(args.outdir, "figures"); os.makedirs(fig_dir, exist_ok=True)

    generic, specific = load_label_map(args.label_map)
    patients = sorted(os.path.splitext(f)[0] for f in os.listdir(args.data_root)
                      if f.endswith(".csv") and not f.startswith("label_map"))
    records, fractions_all, features_all, norm_all = [], {}, {}, {}
    
    for p in patients:
        try:
            curves = load_patient_csv(p, args.data_root, generic, specific)
            rec, fr, ft, nc = process_patient(p, curves, cfg)
        except Exception as e:
            print(f"[skip] {p}: {e}"); continue
        records.append(rec); fractions_all[p] = fr; features_all[p] = ft; norm_all[p] = nc
        
    if not records:
        print("Nessun paziente elaborato (controlla label_map e CSV)."); sys.exit(1)

    frac_df, terr = aggregate(records, fractions_all, cfg)
    stats_out = run_statistics(terr, cfg)

    # Salvataggio CSV dei risultati
    frac_df.to_csv(os.path.join(args.outdir, "frazioni_per_paziente.csv"))
    terr.to_csv(os.path.join(args.outdir, "territori_per_paziente.csv"))
    terr.agg(["mean", "std", "min", "max"]).T.to_csv(os.path.join(args.outdir, "riepilogo_territori.csv"))
    stats_out["level2_thesis"].to_csv(os.path.join(args.outdir, "livello2_vs_tesi.csv"), index=False)
    
    feat_rows = [{"patient": p, "outlet": lab, **fv}
                 for p, fd in features_all.items() for lab, fv in fd.items()]
    pd.DataFrame(feat_rows).to_csv(os.path.join(args.outdir, "feature_forma.csv"), index=False)

    shape_df = shape_metrics_vs_les(norm_all, cfg)
    shape_df.to_csv(os.path.join(args.outdir, "forma_vs_les_per_paziente.csv"), index=False)

    # Generazione Figure e Report
    spaghetti_per_outlet(norm_all, cfg, fig_dir)
    aggregate_vs_les(norm_all, cfg, fig_dir, shape_df)
    fractions_summary(terr, stats_out, cfg, fig_dir)
    write_report(records, terr, stats_out, features_all, cfg, args.outdir)
    print(f"\n[Analisi Completata] Output salvati in: {args.outdir}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    ap = argparse.ArgumentParser(description="Confronto flusso outlet CFD vs letteratura (Ver. 3)")
    sub = ap.add_subparsers(dest="mode", required=True)

    pe = sub.add_parser("export", help="estrai dai VTU -> CSV + template etichette")
    pe.add_argument("--export-dir", default="./sim_csv")

    pa = sub.add_parser("analyze", help="CSV + label_map -> statistica e figure")
    pa.add_argument("--data-root", default="./sim_csv")
    pa.add_argument("--label-map", default="./sim_csv/label_map.csv")
    pa.add_argument("--outdir", default="./results")
    pa.add_argument("--period", type=float, default=0.8)
    pa.add_argument("--tost-margin", type=float, default=DEFAULT_TOST_MARGIN,
                    help="Margine TOST (default: 0.5 * SD_Les ~ 0.042)")
    pa.add_argument("--shape-norm", choices=["mean", "pulse"], default="mean",
                    help="'mean'=divide per la media (Q/Q_mean, default fisiologico); 'pulse'=demean+ampiezza")
    pa.add_argument("--no-align", action="store_true", help="disattiva l'allineamento sul picco dell'inlet")

    args = ap.parse_args()
    if args.mode == "export":
        export_mode(args)
    elif args.mode == "analyze":
        analyze_mode(args)


if __name__ == "__main__":
    main()