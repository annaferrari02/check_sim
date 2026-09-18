#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_flow_comparison.py
======================

Confronto della distribuzione di flusso agli outlet (CFD post-EVAR) con la letteratura.
UNICO script, DUE modalita':

  export   legge i VTU tramite il TUO io_vtu (read_faces + extract_timeseries),
           scrive un CSV per paziente (time + flusso GREZZO col segno per ogni
           faccia inlet/outlet) e un TEMPLATE di etichette da compilare a mano.
             --> dipende dal tuo io_vtu/config; NON lo tocca.

  analyze  legge i CSV + la mappa di etichette (compilata da te) e produce:
             frazioni per paziente, statistica (Livello 1 vs Les, Livello 2 vs tesi),
             una figura per outlet con tutte le curve dei pazienti, il confronto di
             forma iliache-vs-Les-IR, i violini delle frazioni e un report.txt.
             --> completamente autonomo (solo numpy/scipy/pandas/matplotlib).

FLUSSO DI LAVORO
----------------
  1) python3 run_flow_comparison.py export  --export-dir ./sim_csv
  2) apri  ./sim_csv/label_map_TEMPLATE.csv , compila le colonne label/group,
     salvalo come  ./sim_csv/label_map.csv
  3) python3 run_flow_comparison.py analyze --data-root ./sim_csv \
         --label-map ./sim_csv/label_map.csv --outdir ./results --period 0.8

 

CONVENZIONI
-----------
  - times in secondi (steps*dt); Q in mL/s col segno (inlet<0). Il flusso viene
    scritto GREZZO: e' analyze a orientare ogni curva a media positiva, cosi' un
    tratto negativo = reverse fisiologico (serve al check di forma).
  - Le frazioni usano i moduli delle medie di ciclo -> robuste, adimensionali.
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
LES_SC_WAVE = np.array([11.8, 13.1, 19.5, 96.1, 204., 203., 172., 136., 103., 60.9,
                        20.2, 4.75, 9.19, 15.0, 18.6, 18.9, 18.4, 18.0, 15.3, 14.7,
                        13.7, 14.0, 13.7, 14.6])
LES_IR_WAVE = np.array([-1.08, -1.80, 2.88, 32.4, 99.6, 114., 92.8, 66.7, 41.1, 15.4,
                        -7.55, -14.2, -10.9, -5.71, -1.67, 0.333, 0.702, 0.594, -0.497,
                        -0.481, -1.17, -0.306, -0.951, 0.418])

# Tesi Surianarayanan 2020 (1 paziente, SENZA SD) - frazione dell'inlet per ramo
THESIS_FRACTION = {"celiac": 0.218, "sma": 0.147, "renal_L": 0.147, "renal_R": 0.147}


# ============================================================================
# CONFIG
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
    cardiac_period: float | None = 0.8      # il TUO ciclo
    tost_margin: float = 0.05               # GIUSTIFICARE fisiologicamente
    alpha: float = 0.05
    reverse_flag_threshold: float = 0.02


# ============================================================================
# MODALITA' EXPORT  (usa il tuo io_vtu)
# ============================================================================

def export_mode(args):
    """Estrae dai VTU e scrive CSV per paziente + template etichette."""
    # >>> se il tuo oggetto config non e' importabile cosi', cambia SOLO queste 2 righe
    import io_vtu
    from config import Config           # <-- adatta se il tuo cfg si costruisce diversamente
    cfg= Config()
    os.makedirs(args.export_dir, exist_ok=True)
    patient_dirs = io_vtu.list_patients(cfg)
    if not patient_dirs:
        print("Nessun paziente trovato con la config attuale."); sys.exit(1)

    names_info = {}   # raw_outlet -> dict(kind, radii[], count)
    for pdir in patient_dirs:
        patient = os.path.basename(os.path.normpath(pdir))
        print(f"[export] {patient}")
        faces = io_vtu.read_faces(pdir, cfg)
        times, P, Q, match_d = io_vtu.extract_timeseries(pdir, faces, cfg)
        print(f"    match {match_d:.1e} | {len(times)} steps")

        cols = {"time": times}
        for f in faces.values():
            if f.kind not in ("inlet", "outlet"):
                continue
            cols[f.name] = Q[f.name]                 # GREZZO, col segno
            info = names_info.setdefault(
                f.name, {"kind": f.kind, "radii": [], "count": 0})
            info["radii"].append(f.radius)
            info["count"] += 1
            c = f.points.mean(axis=0)                # centroide per orientarsi su L/R
            info["centroid"] = (float(c[0]), float(c[1]), float(c[2]))
        pd.DataFrame(cols).to_csv(os.path.join(args.export_dir, f"{patient}.csv"),
                                  index=False)

    # template etichette: una riga per NOME faccia unico (etichetta una volta sola
    # se i nomi sono coerenti tra pazienti; altrimenti aggiungi righe con 'patient').
    rows = []
    for name, info in sorted(names_info.items()):
        rows.append({
            "raw_outlet": name,
            "kind": info["kind"],
            "n_patients": info["count"],
            "median_radius_cm": float(np.median(info["radii"])),
            "cx": info["centroid"][0], "cy": info["centroid"][1], "cz": info["centroid"][2],
            "label": "inlet" if info["kind"] == "inlet" else "",   # <-- COMPILA
            "group": "inlet" if info["kind"] == "inlet" else "",   # visceral_renal | iliac
        })
    tpl = os.path.join(args.export_dir, "label_map_TEMPLATE.csv")
    pd.DataFrame(rows).to_csv(tpl, index=False)

    print("\n" + "=" * 66)
    print(f"CSV scritti in {args.export_dir}")
    print(f"TEMPLATE etichette: {tpl}")
    print("Compila 'label' (celiac|sma|renal_L|renal_R|iliac_L|iliac_R) e 'group'")
    print("(visceral_renal|iliac), usa median_radius_cm e cx/cy/cz per L/R,")
    print("salva come label_map.csv, poi lancia la modalita' analyze.")
    print("=" * 66)


# ============================================================================
# CARICAMENTO PER ANALYZE
# ============================================================================

def load_label_map(path):
    """CSV: raw_outlet,label,group [,patient].
    Se 'patient' e' presente in una riga, quella vince per quel paziente."""
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
    """CSV <data_root>/<patient>.csv -> dict label -> (t, Q). Somma piu' colonne
    che mappano alla stessa label (es. hepatic+splenic -> celiac)."""
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
        if lab in curves:                    # somma (stessa label da piu' facce)
            curves[lab] = (t, curves[lab][1] + Q)
        else:
            curves[lab] = (t, Q)
    return curves


# ============================================================================
# ELABORAZIONE (identica alla versione testata)
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


def shape_features(t, Q, mean_flow):
    T = t[-1] - t[0]
    mf = abs(mean_flow) if abs(mean_flow) > 1e-12 else 1e-12
    pos, neg = np.clip(Q, 0, None), np.clip(Q, None, 0)
    fwd = float(np.trapezoid(pos, t)); rev = float(-np.trapezoid(neg, t))
    tn = (t - t[0]) / T if T > 0 else t
    dia = Q[tn >= 0.5]
    return dict(
        pulsatility=(float(np.max(Q)) - float(np.min(Q))) / mf,
        rev_vol_frac=(rev / fwd if fwd > 1e-12 else np.nan),
        rev_time_frac=float(np.mean(Q < 0)),
        t_peak=(float((t[np.argmax(Q)] - t[0]) / T) if T > 0 else np.nan),
        dia_level=(float(np.mean(dia) / mf) if dia.size else np.nan),
    )


def process_patient(patient, curves, cfg):
    if cfg.inlet_label not in curves:
        raise ValueError(f"{patient}: manca l'inlet '{cfg.inlet_label}'")
    ti, Qi = extract_last_cycle(*curves[cfg.inlet_label], cfg.cardiac_period)
    Qi_o, _ = orient_positive(ti, Qi)
    inlet_mean = cycle_mean(ti, Qi_o)

    rec = {"patient": patient, "inlet_mean": inlet_mean}
    fractions, features, norm_curves = {}, {}, {}
    outlet_sum = 0.0
    for label, (t, Q) in curves.items():
        if label == cfg.inlet_label:
            continue
        tt, QQ = extract_last_cycle(t, Q, cfg.cardiac_period)
        QQ_o, _ = orient_positive(tt, QQ)
        m = cycle_mean(tt, QQ_o)
        outlet_sum += abs(m)
        fractions[label] = abs(m) / abs(inlet_mean) if abs(inlet_mean) > 1e-12 else np.nan
        features[label] = shape_features(tt, QQ_o, m)
        _, Qn = resample_cycle(tt, QQ_o, cfg.n_resample)
        norm_curves[label] = Qn / m if abs(m) > 1e-12 else Qn

    rec["mass_closure"] = outlet_sum / abs(inlet_mean) if abs(inlet_mean) > 1e-12 else np.nan
    _, Qin = resample_cycle(ti, Qi_o, cfg.n_resample)
    norm_curves[cfg.inlet_label] = Qin / inlet_mean if abs(inlet_mean) > 1e-12 else Qin

    iliac_members = cfg.territories.get("iliac_total", [])
    abs_curves = []
    for label, (t, Q) in curves.items():
        if label in iliac_members:
            tt, QQ = extract_last_cycle(t, Q, cfg.cardiac_period)
            QQ_o, _ = orient_positive(tt, QQ)
            _, Qres = resample_cycle(tt, QQ_o, cfg.n_resample)
            abs_curves.append(Qres)
    if abs_curves:
        agg = np.sum(np.vstack(abs_curves), axis=0)
        magg = float(np.mean(agg))
        norm_curves["iliac_total"] = agg / magg if abs(magg) > 1e-12 else agg
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
# STATISTICA
# ============================================================================

def tost_one_sample(x, target, margin, alpha=0.05):
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
# FIGURE
# ============================================================================

def spaghetti_per_outlet(norm_all, cfg, outdir):
    os.makedirs(outdir, exist_ok=True)
    grid = np.linspace(0, 1, cfg.n_resample)
    labels = sorted({l for c in norm_all.values() for l in c if l != cfg.inlet_label})
    for label in labels:
        curves = [c[label] for c in norm_all.values() if label in c]
        if not curves:
            continue
        M = np.vstack(curves)
        mean_c, sd_c = np.nanmean(M, axis=0), np.nanstd(M, axis=0)
        fig, ax = plt.subplots(figsize=(7, 4.3))
        for row in M:
            ax.plot(grid, row, color="0.72", lw=0.7, alpha=0.55)
        ax.plot(grid, mean_c, color="C0", lw=2.4, label=f"media coorte (n={M.shape[0]})")
        ax.fill_between(grid, mean_c - sd_c, mean_c + sd_c, color="C0", alpha=0.18, label="±1 SD")
        if label == "iliac_total":
            _, ir = resample_cycle(np.linspace(0, 1, len(LES_IR_WAVE)), LES_IR_WAVE, cfg.n_resample)
            ax.plot(grid, ir / np.mean(LES_IR_WAVE), color="C3", lw=2.2, ls="--", label="Les IR (rif.)")
        ax.axhline(0, color="k", lw=0.8, ls=":")
        ax.set_title(f"Curve di flusso normalizzate - {label}")
        ax.set_xlabel("fase del ciclo cardiaco [-]"); ax.set_ylabel("flusso / media di ciclo [-]")
        ax.legend(fontsize=8, loc="upper right"); ax.grid(alpha=0.25)
        fig.tight_layout(); fig.savefig(os.path.join(outdir, f"curve_{label}.png"), dpi=140)
        plt.close(fig)


def distal_vs_les(norm_all, cfg, outdir):
    grid = np.linspace(0, 1, cfg.n_resample)
    curves = [c["iliac_total"] for c in norm_all.values() if "iliac_total" in c]
    if not curves:
        return
    M = np.vstack(curves); mean_c, sd_c = np.nanmean(M, axis=0), np.nanstd(M, axis=0)
    _, ir = resample_cycle(np.linspace(0, 1, len(LES_IR_WAVE)), LES_IR_WAVE, cfg.n_resample)
    fig, ax = plt.subplots(figsize=(7, 4.3))
    ax.plot(grid, mean_c, color="C0", lw=2.4, label="iliache L+R (media coorte)")
    ax.fill_between(grid, mean_c - sd_c, mean_c + sd_c, color="C0", alpha=0.18, label="±1 SD")
    ax.plot(grid, ir / np.mean(LES_IR_WAVE), color="C3", lw=2.2, ls="--", label="Les IR (36 AAA)")
    ax.axhline(0, color="k", lw=0.8, ls=":")
    ax.set_title("Forma aggregato distale vs Les IR (normalizzate)")
    ax.set_xlabel("fase del ciclo [-]"); ax.set_ylabel("flusso / media [-]")
    ax.legend(fontsize=8); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "forma_distale_vs_les.png"), dpi=140)
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
# REPORT
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
        A(f"  TOST (±{cfg.tost_margin}): p={t['p_tost']:.4f} -> "
          f"{'EQUIVALENTE' if t['equivalent'] else 'NON equivalente'}")
        A(f"  Welch: diff={w['diff']:+.3f} (IC95 [{w['ci95'][0]:+.3f},{w['ci95'][1]:+.3f}]), "
          f"p={w['p']:.4f}, d={w['cohens_d']:+.2f}")
    if "level1_visceral_renal" in stats_out:
        s = stats_out["level1_visceral_renal"]; t = s["tost"]
        A(f"\n  viscere+reni vs {s['ref_mean']:.3f}: media {t['mean']:.3f} ± {t['sd']:.3f}, "
          f"coverage {s['coverage']*100:.1f}%, TOST p={t['p_tost']:.4f} "
          f"({'EQUIVALENTE' if t['equivalent'] else 'NON equiv.'})")
    A("\n" + "-" * 70)
    A("LIVELLO 2 (descrittivo) - split interno vs tesi (1 pz, NO SD): solo scarto")
    A("-" * 70)
    for _, r in stats_out["level2_thesis"].iterrows():
        A(f"  {r['ramo']:9s}: coorte {r['coorte_mean']:.3f} ± {r['coorte_sd']:.3f} | "
          f"tesi {r['tesi_ref']:.3f} | scarto {r['scarto_abs']:+.3f} ({r['scarto_rel']*100:+.1f}%)")
    A("\n" + "-" * 70)
    A(f"DIAGNOSTICA - reverse diastolico renale (soglia {cfg.reverse_flag_threshold})")
    A("-" * 70)
    for rlabel in ["renal_L", "renal_R"]:
        vals = np.array([f[rlabel]["rev_time_frac"] for f in features_all.values() if rlabel in f], float)
        if vals.size:
            A(f"  {rlabel}: reverse medio {np.nanmean(vals):.3f}; "
              f"{int(np.sum(vals > cfg.reverse_flag_threshold))}/{vals.size} pz sopra soglia "
              f"(possibile artefatto RCR: res1_split uniforme).")
    rep = "\n".join(L)
    with open(os.path.join(outdir, "report.txt"), "w") as fh:
        fh.write(rep + "\n")
    print(rep)


# ============================================================================
# ANALYZE + DEMO
# ============================================================================

def analyze_mode(args):
    cfg = Config(cardiac_period=args.period, tost_margin=args.tost_margin)
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

    frac_df.to_csv(os.path.join(args.outdir, "frazioni_per_paziente.csv"))
    terr.to_csv(os.path.join(args.outdir, "territori_per_paziente.csv"))
    terr.agg(["mean", "std", "min", "max"]).T.to_csv(os.path.join(args.outdir, "riepilogo_territori.csv"))
    stats_out["level2_thesis"].to_csv(os.path.join(args.outdir, "livello2_vs_tesi.csv"), index=False)
    feat_rows = [{"patient": p, "outlet": lab, **fv}
                 for p, fd in features_all.items() for lab, fv in fd.items()]
    pd.DataFrame(feat_rows).to_csv(os.path.join(args.outdir, "feature_forma.csv"), index=False)

    spaghetti_per_outlet(norm_all, cfg, fig_dir)
    distal_vs_les(norm_all, cfg, fig_dir)
    fractions_summary(terr, stats_out, cfg, fig_dir)
    write_report(records, terr, stats_out, features_all, cfg, args.outdir)
    print(f"\nOutput in: {args.outdir}")


def demo_mode(args):
    """Genera CSV + label_map SINTETICI (finti) e lancia analyze."""
    rng = np.random.default_rng(0)
    ddir = os.path.join(args.outdir, "_demo_csv"); os.makedirs(ddir, exist_ok=True)
    names = ["inlet", "celiac", "sma", "renal_L", "renal_R", "iliac_L", "iliac_R"]
    tt = np.linspace(0, 1, 200)
    sc = np.interp(tt, np.linspace(0, 1, len(LES_SC_WAVE)), LES_SC_WAVE); sc /= sc.mean()
    ir = np.interp(tt, np.linspace(0, 1, len(LES_IR_WAVE)), LES_IR_WAVE); ir /= ir.mean()
    visc = 0.85 * sc + 0.15 * np.clip(ir, 0, None); visc /= visc.mean()
    for i in range(42):
        p = f"pz{100+i:03d}"; im = rng.normal(51.2, 8.0); t = np.linspace(0, 0.8, 200)
        fr = {"celiac": rng.normal(.21, .03), "sma": rng.normal(.15, .025),
              "renal_L": rng.normal(.135, .03), "renal_R": rng.normal(.135, .03)}
        fr = {k: max(v, .02) for k, v in fr.items()}
        il = max(1 - sum(fr.values()), .10); sp = rng.uniform(.45, .55)
        cols = {"time": t, "inlet": -im * sc}
        for k in ["celiac", "sma"]:
            cols[k] = im * fr[k] * visc * rng.normal(1, .03, t.size)
        for k in ["renal_L", "renal_R"]:
            base = 0.7 * visc + 0.3 * ir if rng.random() < .25 else visc
            cols[k] = im * fr[k] * base * rng.normal(1, .03, t.size)
        cols["iliac_L"] = im * il * sp * ir * rng.normal(1, .03, t.size)
        cols["iliac_R"] = im * il * (1 - sp) * ir * rng.normal(1, .03, t.size)
        pd.DataFrame(cols).to_csv(os.path.join(ddir, f"{p}.csv"), index=False)
    lm = os.path.join(ddir, "label_map.csv")
    pd.DataFrame([{"raw_outlet": n, "label": n,
                   "group": "inlet" if n == "inlet" else ("iliac" if n.startswith("iliac") else "visceral_renal")}
                  for n in names]).to_csv(lm, index=False)
    print(f"[demo] dati SINTETICI in {ddir}")
    args.data_root = ddir; args.label_map = lm; args.period = 0.8
    analyze_mode(args)


# ============================================================================
# MAIN
# ============================================================================

def main():
    ap = argparse.ArgumentParser(description="Confronto flusso outlet CFD vs letteratura")
    sub = ap.add_subparsers(dest="mode", required=True)

    pe = sub.add_parser("export", help="estrai dai VTU -> CSV + template etichette")
    pe.add_argument("--export-dir", default="./sim_csv")

    pa = sub.add_parser("analyze", help="CSV + label_map -> statistica e figure")
    pa.add_argument("--data-root", default="./sim_csv")
    pa.add_argument("--label-map", default="./sim_csv/label_map.csv")
    pa.add_argument("--outdir", default="./results")
    pa.add_argument("--period", type=float, default=0.8)
    pa.add_argument("--tost-margin", type=float, default=0.05)


    args = ap.parse_args()
    if args.mode == "export":
        export_mode(args)
    elif args.mode == "analyze":
        analyze_mode(args)
    elif args.mode == "demo":
        demo_mode(args)


if __name__ == "__main__":
    main()