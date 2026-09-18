#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_flow_comparison.py
======================

Confronto della distribuzione di flusso agli outlet (CFD post-EVAR) con la
letteratura. UNICO script, DUE modalita':

  export   legge i VTU tramite il TUO io_vtu (read_faces + extract_timeseries),
           scrive un CSV per paziente (time + flusso GREZZO col segno per ogni
           faccia inlet/outlet) e un TEMPLATE di etichette da compilare a mano.
             --> dipende dal tuo io_vtu/config; NON lo tocca.

  analyze  legge i CSV + la mappa di etichette (compilata da te) e produce:
             frazioni per paziente, statistica (Livello 1 vs Les, Livello 2 vs
             tesi), una figura per outlet con tutte le curve dei pazienti, il
             confronto di forma iliache-vs-Les-IR, i violini delle frazioni e
             un report.txt.
             --> completamente autonomo (solo numpy/scipy/pandas/matplotlib).

COSA VIENE REALMENTE CONFRONTATO (leggere prima di interpretare i numeri)
------------------------------------------------------------------------
  * LIVELLO 1 / 2 = confronto sulle MEDIE DI CICLO (frazioni di portata). Con
    BC di tipo RCR impostate via Murray (r^3, vedi setup_bcs), la ripartizione
    media del flusso e' in larga parte PRESCRITTA dal disegno delle resistenze,
    non emergente dalla CFD: uno scostamento sistematico dalla letteratura e'
    quindi anzitutto un'affermazione su geometria (raggi outlet) + disegno BC,
    non sul solutore. Vedi note_metodologiche.md, sezione "framing per il prof".
  * FORMA = confronto sulla SOLA forma d'onda (mean rimossa se shape_norm=pulse
    e picchi allineati). Risponde a una domanda diversa: la pulsatilita' e' quella
    giusta? Puo' concordare anche quando le frazioni no, e viceversa. Tenere i due
    piani SEPARATI nel commento dei risultati.

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
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")               # backend non interattivo: salva PNG, non apre finestre
import matplotlib.pyplot as plt
from scipy import stats

# numpy>=2.0 rinomina np.trapz -> np.trapezoid. Shim per girare su entrambe
# le versioni senza cambiare il resto del codice.
_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))


# ============================================================================
# RIFERIMENTI DI LETTERATURA
# ============================================================================
# NB: tutti i valori qui sotto sono HARD-CODED e vanno trattati come "verita' di
# riferimento". Due sorgenti, con incertezza molto diversa: Les (coorte n=36, ha
# media+SD) e tesi Surianarayanan (1 paziente, NIENTE SD). Questo determina che
# statistica e' lecita a valle (vedi Livello 1 vs Livello 2).

# Les et al., Cardiovasc Eng Technol 2010 (coorte modello n=36).
LES_N = 36
# Rapporto di portata media infrarenale/supraceliaca (IR/SC). E' il riferimento
# a cui confrontiamo la frazione iliaca totale: per conservazione di massa, in un
# modello i cui outlet a valle dei reni sono solo le iliache, il flusso medio che
# raggiunge le iliache == flusso infrarenale. Quindi frazione_iliache ~ IR/SC.
LES_IRSC_RATIO_MEAN = 0.343
LES_IRSC_RATIO_SD = 0.0843
# Waveform SUPRACELIACA (SC) digitalizzata dalla figura di Les (24 campioni).
LES_SC_WAVE = np.array([11.8, 13.1, 19.5, 96.1, 204., 203., 172., 136., 103., 60.9,
                        20.2, 4.75, 9.19, 15.0, 18.6, 18.9, 18.4, 18.0, 15.3, 14.7,
                        13.7, 14.0, 13.7, 14.6])
# Waveform INFRARENALE (IR) digitalizzata da Les (24 campioni). Ha reverse
# diastolico marcato (valori negativi) -> serve da riferimento di FORMA iliaca.
LES_IR_WAVE = np.array([-1.08, -1.80, 2.88, 32.4, 99.6, 114., 92.8, 66.7, 41.1, 15.4,
                        -7.55, -14.2, -10.9, -5.71, -1.67, 0.333, 0.702, 0.594, -0.497,
                        -0.481, -1.17, -0.306, -0.951, 0.418])
# Aggregato viscere+reni = SC - IR, per conservazione di massa frame-per-frame.
# ATTENZIONE (assunzione forte): sottrarre campione-per-campione presuppone che
# le due waveform digitalizzate siano GIA' in fase (stesso riferimento temporale).
# Se SC e IR provengono da figure/soggetti con fase diversa, LES_VR_WAVE e' distorta.
# --> GIUSTIFICARE / verificare in fase di digitalizzazione. Vedi note_metodologiche.
LES_VR_WAVE = LES_SC_WAVE - LES_IR_WAVE

# Riferimenti di FORMA disponibili (aggregati con waveform di letteratura).
# inlet<->SC e' un sanity check sulla BC (la portata inlet e' IMPOSTA, non emergente):
# se la forma inlet non combacia con SC, il problema e' a monte (inflow), non nei rami.
REF_WAVES = {"inlet": LES_SC_WAVE, "visceral_renal": LES_VR_WAVE, "iliac_total": LES_IR_WAVE}
REF_LABEL = {"inlet": "Les SC", "visceral_renal": "Les SC-IR", "iliac_total": "Les IR"}
VR_MEMBERS = ["celiac", "sma", "renal_L", "renal_R"]   # rami che compongono viscere+reni

# Tesi Surianarayanan 2020 (1 paziente, SENZA SD) - frazione dell'inlet per ramo.
# Usata SOLO al Livello 2 come confronto descrittivo (scarto), mai per test.
THESIS_FRACTION = {"celiac": 0.218, "sma": 0.147, "renal_L": 0.147, "renal_R": 0.147}


# ============================================================================
# CONFIG
# ============================================================================

@dataclass
class Config:
    # Come i nomi-faccia grezzi (label_map) si raggruppano in territori vascolari.
    # 'iliac_total' assorbe qualunque variante di nome delle iliache (comune/est/int, L/R).
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
    n_resample: int = 100                   # campioni per ciclo dopo ricampionamento uniforme
    # PERIODO CARDIACO usato per ritagliare l'ultimo ciclo. VERIFICARE che coincida
    # col periodo con cui hai girato la sim: setup_bcs normalizza l'inflow a 1.0 s
    # (t[-1]->1). Se la sim gira a 1.0 s e qui usi 0.8 s, extract_last_cycle ritaglia
    # la finestra SBAGLIATA -> medie di ciclo e frazioni falsate. --> VERIFICA PRIORITARIA.
    cardiac_period: float | None = 0.8
    tost_margin: float = 0.05               # margine di equivalenza TOST. --> GIUSTIFICARE fisiologicamente
    alpha: float = 0.05
    reverse_flag_threshold: float = 0.02    # soglia diagnostica: frazione di ciclo con Q<0 ai reni
    shape_norm: str = "pulse"               # 'pulse' (demean+ampiezza) | 'mean' (divide-by-mean)
    shape_align: bool = True                # peak-alignment prima del confronto di forma


# ============================================================================
# MODALITA' EXPORT  (usa il tuo io_vtu)
# ============================================================================

def export_mode(args):
    """Estrae dai VTU e scrive CSV per paziente + template etichette.

    Unica parte accoppiata al tuo ambiente (io_vtu/config). Scrive il flusso
    GREZZO col segno: nessuna elaborazione qui, cosi' analyze resta l'unico
    punto in cui si prendono decisioni (orientamento, ciclo, frazioni).
    """
    # >>> se il tuo oggetto config non e' importabile cosi', cambia SOLO queste 2 righe
    import io_vtu
    from config import Config           # <-- NB: questo Config (di config.py) SHADOWA il dataclass qui sopra
    cfg = Config()
    os.makedirs(args.export_dir, exist_ok=True)
    patient_dirs = io_vtu.list_patients(cfg)
    if not patient_dirs:
        print("Nessun paziente trovato con la config attuale."); sys.exit(1)

    names_info = {}   # raw_outlet -> dict(kind, radii[], count, centroid): serve al template etichette
    for pdir in patient_dirs:
        patient = os.path.basename(os.path.normpath(pdir))
        print(f"[export] {patient}")
        faces = io_vtu.read_faces(pdir, cfg)
        # extract_timeseries restituisce anche match_d = distanza di matching faccia<->serie
        # (quanto bene le facce nominate combaciano coi dati): utile come QC di lettura.
        times, P, Q, match_d = io_vtu.extract_timeseries(pdir, faces, cfg)
        print(f"    match {match_d:.1e} | {len(times)} steps")

        cols = {"time": times}
        for f in faces.values():
            if f.kind not in ("inlet", "outlet"):
                continue                                 # pareti/altro non entrano nel CSV
            cols[f.name] = Q[f.name]                      # GREZZO, col segno
            # Accumula metadati per faccia (mediati/contati sui pazienti) per il template.
            info = names_info.setdefault(
                f.name, {"kind": f.kind, "radii": [], "count": 0})
            info["radii"].append(f.radius)
            info["count"] += 1
            c = f.points.mean(axis=0)                    # centroide: aiuta a distinguere L/R nel template
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
            "median_radius_cm": float(np.median(info["radii"])),   # raggio -> aiuta a mappare il ramo
            "cx": info["centroid"][0], "cy": info["centroid"][1], "cz": info["centroid"][2],
            "label": "inlet" if info["kind"] == "inlet" else "",   # <-- COMPILA a mano
            "group": "inlet" if info["kind"] == "inlet" else "",   # visceral_renal | iliac  <-- COMPILA
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
    Se 'patient' e' presente in una riga, quella vince per quel paziente
    (override specifico > mapping generico)."""
    df = pd.read_csv(path)
    if not {"raw_outlet", "label"}.issubset(df.columns):
        raise ValueError("label_map deve avere almeno colonne raw_outlet,label")
    df = df[df["label"].astype(str).str.strip() != ""]           # scarta righe non etichettate
    # mapping generico: vale per tutti i pazienti (righe senza 'patient')
    generic = {r["raw_outlet"]: r["label"] for _, r in df.iterrows()
               if "patient" not in df.columns or pd.isna(r.get("patient"))}
    # mapping specifico: (paziente, faccia_grezza) -> label, per gestire eccezioni per-paziente
    specific = {}
    if "patient" in df.columns:
        for _, r in df.iterrows():
            if not pd.isna(r.get("patient")):
                specific[(str(r["patient"]), r["raw_outlet"])] = r["label"]
    return generic, specific


def resolve_label(patient, raw, generic, specific):
    """Override specifico se esiste, altrimenti mapping generico (None se non mappato)."""
    if (patient, raw) in specific:
        return specific[(patient, raw)]
    return generic.get(raw)


def load_patient_csv(patient, data_root, generic, specific):
    """CSV <data_root>/<patient>.csv -> dict label -> (t, Q). Somma piu' colonne
    che mappano alla stessa label (es. hepatic+splenic -> celiac).

    La somma e' sui flussi GREZZI col segno: corretta perche' rami dello stesso
    territorio hanno lo stesso verso (l'orientamento a media positiva avviene dopo)."""
    df = pd.read_csv(os.path.join(data_root, f"{patient}.csv"))
    tcol = "time" if "time" in df.columns else df.columns[0]
    t = df[tcol].to_numpy(float)
    curves = {}
    for col in df.columns:
        if col == tcol:
            continue
        lab = resolve_label(patient, col, generic, specific)
        if lab is None:
            continue                              # colonna non mappata -> ignorata
        Q = df[col].to_numpy(float)
        if lab in curves:                         # stessa label da piu' facce -> somma
            curves[lab] = (t, curves[lab][1] + Q)
        else:
            curves[lab] = (t, Q)
    return curves


# ============================================================================
# ELABORAZIONE (identica alla versione testata)
# ============================================================================

def extract_last_cycle(t, Q, period):
    """Ritaglia l'ULTIMO ciclo cardiaco (per evitare il transitorio iniziale) e
    ri-azzera il tempo. Se period e' None o piu' lungo della serie, prende tutto.

    ATTENZIONE: la finestra e' [t[-1]-period, t[-1]]. Se 'period' non coincide col
    periodo REALE della simulazione, la media di ciclo calcolata a valle e' su una
    finestra sbagliata. --> vedi Config.cardiac_period."""
    if period is None or period >= (t[-1] - t[0]):
        return t - t[0], Q
    mask = t >= (t[-1] - period - 1e-12)          # -1e-12: tolleranza per il campione di bordo
    tt, QQ = t[mask], Q[mask]
    return tt - tt[0], QQ


def cycle_mean(t, Q):
    """Media di ciclo pesata sul tempo (integrale trapezoidale / durata).
    Non e' la media aritmetica dei campioni: e' robusta a passi dt non uniformi."""
    T = t[-1] - t[0]
    return float(np.mean(Q)) if T <= 0 else float(_trapz(Q, t) / T)


def orient_positive(t, Q):
    """Orienta la curva a media di ciclo positiva. Serve perche' SimVascular usa
    il segno rispetto alla normale uscente della faccia: l'inlet e' negativo, e i
    singoli outlet possono avere convenzioni diverse. Ritorna (Q_orientata, flipped)."""
    return (-Q, True) if cycle_mean(t, Q) < 0 else (Q, False)


def resample_cycle(t, Q, n):
    """Ricampiona su n punti in fase normalizzata [0,1] (interp lineare). Rende
    confrontabili cicli con numero di step diverso tra pazienti."""
    tn = (t - t[0]) / (t[-1] - t[0])
    grid = np.linspace(0.0, 1.0, n)
    return grid, np.interp(grid, tn, Q)


def shape_features(t, Q, mean_flow):
    """Metriche scalari di forma d'onda, tutte adimensionali (normalizzate su |media|):
      - pulsatility : (max-min)/|media|            -> ampiezza relativa del polso
      - rev_vol_frac: volume retrogrado/anterogrado -> quanto flusso torna indietro
      - rev_time_frac: frazione di ciclo con Q<0     -> per quanto tempo c'e' reverse
      - t_peak      : fase [0,1] del picco sistolico
      - dia_level   : livello medio in diastole (fase>=0.5), normalizzato su |media|
    """
    T = t[-1] - t[0]
    mf = abs(mean_flow) if abs(mean_flow) > 1e-12 else 1e-12       # guardia anti-div0
    pos, neg = np.clip(Q, 0, None), np.clip(Q, None, 0)            # parte anterograda / retrograda
    fwd = float(_trapz(pos, t)); rev = float(-_trapz(neg, t))       # volumi (rev reso positivo)
    tn = (t - t[0]) / T if T > 0 else t
    dia = Q[tn >= 0.5]                                             # seconda meta' del ciclo ~ diastole
    return dict(
        pulsatility=(float(np.max(Q)) - float(np.min(Q))) / mf,
        rev_vol_frac=(rev / fwd if fwd > 1e-12 else np.nan),
        rev_time_frac=float(np.mean(Q < 0)),
        t_peak=(float((t[np.argmax(Q)] - t[0]) / T) if T > 0 else np.nan),
        dia_level=(float(np.mean(dia) / mf) if dia.size else np.nan),
    )


def process_patient(patient, curves, cfg):
    """Cuore dell'analisi per singolo paziente. Restituisce:
      rec   : dict con inlet_mean e mass_closure (QC)
      fractions: label -> |media outlet| / |media inlet|   (adimensionale)
      features : label -> metriche di forma
      norm_curves: label -> curva orientata ricampionata (per le figure di forma)
    """
    if cfg.inlet_label not in curves:
        raise ValueError(f"{patient}: manca l'inlet '{cfg.inlet_label}'")
    # --- INLET: ultimo ciclo, orientato positivo, media di ciclo ---
    ti, Qi = extract_last_cycle(*curves[cfg.inlet_label], cfg.cardiac_period)
    Qi_o, _ = orient_positive(ti, Qi)
    inlet_mean = cycle_mean(ti, Qi_o)                              # denominatore di tutte le frazioni

    rec = {"patient": patient, "inlet_mean": inlet_mean}
    fractions, features, norm_curves = {}, {}, {}
    outlet_sum = 0.0
    # --- OGNI OUTLET: frazione, forma, curva ricampionata ---
    for label, (t, Q) in curves.items():
        if label == cfg.inlet_label:
            continue
        tt, QQ = extract_last_cycle(t, Q, cfg.cardiac_period)
        QQ_o, _ = orient_positive(tt, QQ)
        m = cycle_mean(tt, QQ_o)
        outlet_sum += abs(m)                                       # per la chiusura di massa
        # frazione = |media outlet| / |media inlet|. Modulo: adimensionale e robusta al segno.
        fractions[label] = abs(m) / abs(inlet_mean) if abs(inlet_mean) > 1e-12 else np.nan
        features[label] = shape_features(tt, QQ_o, m)
        _, Qn = resample_cycle(tt, QQ_o, cfg.n_resample)
        norm_curves[label] = Qn        # curva orientata GREZZA; la normalizzazione avviene al plot

    # Chiusura di massa: somma dei moduli outlet / |inlet|. Deve valere ~1
    # (conservazione della massa). Scostamenti >2% = problema di lettura/BC. QC chiave.
    rec["mass_closure"] = outlet_sum / abs(inlet_mean) if abs(inlet_mean) > 1e-12 else np.nan
    _, Qin = resample_cycle(ti, Qi_o, cfg.n_resample)
    norm_curves[cfg.inlet_label] = Qin

    # --- AGGREGATI DI FORMA: somma delle curve orientate dei membri ---
    # Servono per confrontare la forma di 'iliac_total' e 'visceral_renal' con Les.
    # Nota: qui si sommano le curve ricampionate GIA' orientate positive (ognuna).
    iliac_members = cfg.territories.get("iliac_total", [])
    for agg_label, members in [("iliac_total", iliac_members), ("visceral_renal", VR_MEMBERS)]:
        abs_curves = []
        for label, (t, Q) in curves.items():
            if label in members:
                tt, QQ = extract_last_cycle(t, Q, cfg.cardiac_period)
                QQ_o, _ = orient_positive(tt, QQ)
                _, Qres = resample_cycle(tt, QQ_o, cfg.n_resample)
                abs_curves.append(Qres)
        if abs_curves:
            norm_curves[agg_label] = np.sum(np.vstack(abs_curves), axis=0)
    return rec, fractions, features, norm_curves


def aggregate(records, fractions_all, cfg):
    """Impila le frazioni per-paziente in due tabelle:
      frac_df: una colonna per label (ramo singolo)
      terr   : una colonna per territorio (somma dei membri), piu' 'visceral_renal'.
    min_count=1 in sum(): un territorio resta NaN solo se TUTTI i membri mancano."""
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
    """TOST a un campione (test di EQUIVALENZA, non di differenza).

    Ipotesi: la media della coorte e' dentro [target-margin, target+margin]?
    Due test one-sided:
      p_low: la media e' significativamente > target-margin ?
      p_up : la media e' significativamente < target+margin ?
    equivalente se ENTRAMBI rifiutano, cioe' p=max(p_low,p_up) < alpha.
    NB: 'non equivalente' NON prova che siano diversi; con n grande e margine
    piccolo e' facile non concludere equivalenza. Da leggere insieme a Welch+d."""
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    n = len(x); mean = float(np.mean(x)); sd = float(np.std(x, ddof=1))
    se = sd / np.sqrt(n); df = n - 1
    p_low = stats.t.sf((mean - (target - margin)) / se, df)       # H0: media <= target-margin
    p_up = stats.t.cdf((mean - (target + margin)) / se, df)       # H0: media >= target+margin
    p = max(p_low, p_up)
    return dict(n=n, mean=mean, sd=sd, p_tost=p, equivalent=(p < alpha),
                bounds=(target - margin, target + margin))


def welch_from_summary(m1, s1, n1, m2, s2, n2):
    """Welch t-test da statistiche riassuntive (non servono i dati grezzi di Les).
    Restituisce differenza, IC95, p e Cohen's d (dimensione dell'effetto).
    d si legge cosi': ~0.2 piccolo, ~0.5 medio, ~0.8 grande."""
    se = np.sqrt(s1**2 / n1 + s2**2 / n2)
    t = (m1 - m2) / se
    df = se**4 / ((s1**2 / n1)**2 / (n1 - 1) + (s2**2 / n2)**2 / (n2 - 1))   # Welch-Satterthwaite
    p = 2 * stats.t.sf(abs(t), df)
    sp = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))      # SD pooled per d
    tc = stats.t.ppf(0.975, df)
    return dict(t=t, df=df, p=p, cohens_d=(m1 - m2) / sp, diff=m1 - m2,
                ci95=((m1 - m2) - tc * se, (m1 - m2) + tc * se))


def coverage(x, ref_mean, ref_sd):
    """Frazione di pazienti che cade nella banda di normalita' della letteratura
    (media ±1.96 SD ~ intervallo 95%). Metrica intuitiva e non parametrica:
    'quanti dei miei pazienti sono plausibili secondo Les?'."""
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    lo, hi = ref_mean - 1.96 * ref_sd, ref_mean + 1.96 * ref_sd
    return float(np.mean((x >= lo) & (x <= hi))), (lo, hi)


def run_statistics(terr, cfg):
    """Assembla i tre livelli di confronto:
      L1 iliache      vs Les IR/SC (0.343 ± 0.0843): coverage + TOST + Welch
      L1 viscere+reni vs 1 - IR/SC (= 0.657): coverage + TOST (stessa SD)
      L2 split interno vs tesi (1 pz, no SD): SOLO scarto assoluto/relativo, niente test."""
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
        ref = 1.0 - LES_IRSC_RATIO_MEAN                          # complementare per conservazione di massa
        cov, band = coverage(x, ref, LES_IRSC_RATIO_SD)          # riusa la SD di IR/SC (approssimazione)
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
# FORMA: allineamento, normalizzazione, metriche
# ============================================================================

def peak_align(y, target_idx):
    """Sposta ciclicamente la curva (periodica) per portare il picco a target_idx.
    Rimuove differenze di FASE tra sim e riferimento. Scelta forte: se c'e' un
    ritardo sistolico reale, questo lo nasconde. Disattivabile con --no-align."""
    return np.roll(y, int(target_idx) - int(np.argmax(y)))


def normalize_curve(y, mode):
    """'mean'  -> divide per la media (media->1, MANTIENE la componente continua).
       'pulse' -> demean e divide per il picco-picco (media->0, FORMA PURA).
    'pulse' isola la forma ma cancella ogni info sulla portata media: due curve
    con frazioni diverse ma stesso profilo appaiono identiche. Sceglierlo di
    proposito quando si vuole giudicare la sola pulsatilita'."""
    if mode == "mean":
        m = float(np.mean(y))
        return y / m if abs(m) > 1e-12 else y
    # pulse
    y0 = y - float(np.mean(y))
    pp = float(y.max() - y.min())
    return y0 / pp if pp > 1e-12 else y0


def _ref_norm(wave, n, mode, align, target):
    """Waveform di riferimento (Les) ricampionata, (peak-aligned), normalizzata:
    stesso trattamento delle curve sim, cosi' il confronto e' apples-to-apples."""
    _, r = resample_cycle(np.linspace(0, 1, len(wave)), wave, n)
    if align:
        r = peak_align(r, target)
    return normalize_curve(r, mode)


def _prep(y, cfg, target):
    """Prepara una curva sim per il confronto/plot: (peak-align) + normalizzazione."""
    y = peak_align(y, target) if cfg.shape_align else y
    return normalize_curve(y, cfg.shape_norm)


def shape_metrics_vs_les(norm_all, cfg):
    """Per ogni aggregato con riferimento (inlet, visceral_renal, iliac_total) e
    per ogni paziente: correlazione e RMSE vs la relativa curva di Les, dopo
    peak-alignment e stessa normalizzazione. Target comune = picco di Les IR
    (tutte le curve, sim e rif., vengono allineate allo stesso indice di fase)."""
    n = cfg.n_resample
    _, ir_raw = resample_cycle(np.linspace(0, 1, len(LES_IR_WAVE)), LES_IR_WAVE, n)
    target = int(np.argmax(ir_raw))                              # indice di fase di riferimento
    rows = []
    for agg, wave in REF_WAVES.items():
        ref = _ref_norm(wave, n, cfg.shape_norm, cfg.shape_align, target)
        for pat, c in norm_all.items():
            if agg not in c:
                continue
            y = _prep(c[agg], cfg, target)
            # corr: somiglianza di forma (fase/andamento); RMSE: distanza puntuale residua.
            corr = float(np.corrcoef(y, ref)[0, 1]) if np.std(y) > 0 else np.nan
            rmse = float(np.sqrt(np.mean((y - ref) ** 2)))
            rows.append({"aggregato": agg, "patient": pat,
                         "corr_vs_les": corr, "rmse_vs_les": rmse})
    return pd.DataFrame(rows), target


# ============================================================================
# FIGURE
# ============================================================================

def _ylabel(cfg):
    return ("flusso demedato / picco-picco [-]" if cfg.shape_norm == "pulse"
            else "flusso / media di ciclo [-]")


def spaghetti_per_outlet(norm_all, cfg, outdir, target):
    """Una figura per outlet: tutte le curve dei pazienti (grigie) + media coorte
    (±1 SD) + curva di Les se disponibile. Serve a vedere dispersione e outlier."""
    os.makedirs(outdir, exist_ok=True)
    grid = np.linspace(0, 1, cfg.n_resample)
    labels = sorted({l for c in norm_all.values() for l in c if l != cfg.inlet_label})
    for label in labels:
        raw = [c[label] for c in norm_all.values() if label in c]
        if not raw:
            continue
        M = np.vstack([_prep(y, cfg, target) for y in raw])       # stesse trasformazioni delle metriche
        mean_c, sd_c = np.nanmean(M, axis=0), np.nanstd(M, axis=0)
        fig, ax = plt.subplots(figsize=(7, 4.3))
        for row in M:
            ax.plot(grid, row, color="0.72", lw=0.7, alpha=0.55)  # curve singole
        ax.plot(grid, mean_c, color="C0", lw=2.4, label=f"media coorte (n={M.shape[0]})")
        ax.fill_between(grid, mean_c - sd_c, mean_c + sd_c, color="C0", alpha=0.18, label="±1 SD")
        if label in REF_WAVES:
            ref = _ref_norm(REF_WAVES[label], cfg.n_resample, cfg.shape_norm, cfg.shape_align, target)
            ax.plot(grid, ref, color="C3", lw=2.2, ls="--", label=f"{REF_LABEL[label]} (rif.)")
        ax.axhline(0, color="k", lw=0.8, ls=":")
        alg = "peak-aligned" if cfg.shape_align else "non allineate"
        ax.set_title(f"Forma normalizzata - {label}  ({cfg.shape_norm}, {alg})")
        ax.set_xlabel("fase del ciclo cardiaco [-]"); ax.set_ylabel(_ylabel(cfg))
        ax.legend(fontsize=8, loc="upper right"); ax.grid(alpha=0.25)
        fig.tight_layout(); fig.savefig(os.path.join(outdir, f"curve_{label}.png"), dpi=140)
        plt.close(fig)


def three_panel_compare(norm_all, records, cfg, outdir, target):
    """Sovrappone inlet / viscere+reni / iliache in tre normalizzazioni:
    assoluto (mL/s) | solo-media | pulse. Risponde a: 'le curve sono identiche
    o e' un artefatto della normalizzazione?'. Riferimenti Les scalati all'inlet.

    Pannello 1 (assoluto) conserva le MAGNITUDINI -> se qui sim e Les divergono
    ma nei pannelli 2-3 no, la differenza e' di portata/frazione, non di forma."""
    grid = np.linspace(0, 1, cfg.n_resample)
    aggs = [("inlet", "C0"), ("visceral_renal", "C2"), ("iliac_total", "C3")]

    # curve medie ASSOLUTE di coorte (mL/s), peak-aligned
    absmean = {}
    for agg, _ in aggs:
        raw = [peak_align(c[agg], target) for c in norm_all.values() if agg in c]
        if raw:
            absmean[agg] = np.nanmean(np.vstack(raw), axis=0)
    inlet_mean = float(np.nanmean([r["inlet_mean"] for r in records]))
    les_factor = inlet_mean / float(np.mean(LES_SC_WAVE))         # scala Les alla tua magnitudine (solo per il pannello assoluto)

    def les_abs(agg):
        _, r = resample_cycle(np.linspace(0, 1, len(REF_WAVES[agg])), REF_WAVES[agg], cfg.n_resample)
        return peak_align(r, target) * les_factor

    panels = [
        ("assoluto (mL/s)", lambda y: y, "flusso [mL/s]"),
        ("solo-media (÷media)", lambda y: y / np.mean(y), "flusso / media [-]"),
        ("pulse (demean ÷ picco-picco)",
         lambda y: (y - np.mean(y)) / (y.max() - y.min()), "forma pura [-]"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for ax, (title, tf, ylab) in zip(axes, panels):
        for agg, col in aggs:
            if agg not in absmean:
                continue
            ax.plot(grid, tf(absmean[agg]), color=col, lw=2.2, label=agg)              # continua = sim
            ax.plot(grid, tf(les_abs(agg)), color=col, lw=1.6, ls="--", alpha=0.8,      # tratteggiata = Les
                    label=f"{REF_LABEL[agg]}")
        ax.axhline(0, color="k", lw=0.7, ls=":")
        ax.set_title(title); ax.set_xlabel("fase [-]"); ax.set_ylabel(ylab)
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=7, ncol=2, loc="upper right")
    fig.suptitle("Tre normalizzazioni a confronto — continua = sim, tratteggiata = Les",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "confronto_3normalizzazioni.png"), dpi=140)
    plt.close(fig)


def aggregate_vs_les(norm_all, cfg, outdir, target, metrics):
    """Una figura dedicata per ogni aggregato con riferimento di Les, con corr e
    RMSE medi annotati nel riquadro."""
    grid = np.linspace(0, 1, cfg.n_resample)
    for agg, wave in REF_WAVES.items():
        raw = [c[agg] for c in norm_all.values() if agg in c]
        if not raw:
            continue
        M = np.vstack([_prep(y, cfg, target) for y in raw])
        mean_c, sd_c = np.nanmean(M, axis=0), np.nanstd(M, axis=0)
        ref = _ref_norm(wave, cfg.n_resample, cfg.shape_norm, cfg.shape_align, target)
        fig, ax = plt.subplots(figsize=(7, 4.3))
        ax.plot(grid, mean_c, color="C0", lw=2.4, label=f"{agg} (media, n={M.shape[0]})")
        ax.fill_between(grid, mean_c - sd_c, mean_c + sd_c, color="C0", alpha=0.18, label="±1 SD")
        ax.plot(grid, ref, color="C3", lw=2.2, ls="--", label=f"{REF_LABEL[agg]} (36 AAA)")
        ax.axhline(0, color="k", lw=0.8, ls=":")
        sub = metrics[metrics["aggregato"] == agg] if not metrics.empty else metrics
        if not sub.empty:
            ax.text(0.02, 0.97, f"corr={sub['corr_vs_les'].mean():.2f}  "
                                f"RMSE={sub['rmse_vs_les'].mean():.2f}",
                    transform=ax.transAxes, va="top", fontsize=9,
                    bbox=dict(boxstyle="round", fc="white", alpha=0.7))
        alg = "peak-aligned" if cfg.shape_align else "non allineate"
        note = "  [sanity check BC]" if agg == "inlet" else ""
        ax.set_title(f"Forma {agg} vs {REF_LABEL[agg]} ({cfg.shape_norm}, {alg}){note}")
        ax.set_xlabel("fase del ciclo [-]"); ax.set_ylabel(_ylabel(cfg))
        ax.legend(fontsize=8); ax.grid(alpha=0.25)
        fig.tight_layout(); fig.savefig(os.path.join(outdir, f"forma_{agg}_vs_les.png"), dpi=140)
        plt.close(fig)


def fractions_summary(terr, stats_out, cfg, outdir):
    """Violini delle frazioni per ramo/territorio, con i punti-tesi (rombi) e la
    banda di normalita' di Les sulle iliache. Vista d'insieme della distribuzione."""
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
# REPORT  (riscritto per leggibilita': sezioni numerate, legenda, chiavi di lettura)
# ============================================================================

def _fmt_interpretation_l1(s, cfg):
    """Traduce coverage + TOST + Welch (se presente) in una riga di lettura sintetica.
    Regola: differenza REALE se coverage bassa E non-equivalenza E |d| non piccolo."""
    t = s["tost"]
    cov = s["coverage"]
    eq = t["equivalent"]
    d = abs(s["welch"]["cohens_d"]) if "welch" in s else None
    if eq:
        return "coorte compatibile con la letteratura (equivalenza entro margine)."
    if d is not None and d >= 0.8 and cov < 0.5:
        return ("scostamento SISTEMATICO e ampio (coverage bassa, effetto grande): "
                "differenza reale, non rumore campionario -> vedi note_metodologiche.")
    if cov < 0.5:
        return "buona parte della coorte fuori banda: scostamento da approfondire."
    return "quadro intermedio: molti pazienti in banda ma non equivalenza formale."


def write_report(records, terr, stats_out, features_all, cfg, outdir, shape_df=None):
    L = []; A = L.append
    line = "=" * 74
    sub = "-" * 74

    # ---- intestazione + parametri della run (tracciabilita') ----
    A(line)
    A("REPORT — DISTRIBUZIONE DI FLUSSO AGLI OUTLET  (CFD post-EVAR vs letteratura)")
    A(line)
    A(f"generato        : {datetime.now():%Y-%m-%d %H:%M}")
    A(f"pazienti        : {len(records)}")
    A(f"periodo ciclo   : {cfg.cardiac_period} s   (deve coincidere col periodo di simulazione!)")
    A(f"forma           : shape_norm={cfg.shape_norm}, "
      f"{'peak-aligned' if cfg.shape_align else 'NON allineate'}")
    A(f"TOST margin     : ±{cfg.tost_margin}   (margine di equivalenza — da giustificare)")

    # ---- legenda delle metriche (perche' il prof legge senza il codice davanti) ----
    A("\n" + sub)
    A("LEGENDA")
    A(sub)
    A("  frazione      = |media di ciclo dell'outlet| / |media inlet|  (adimensionale)")
    A("  chiusura massa= Σ|medie outlet| / |media inlet| ; ideale = 1.000")
    A("  coverage      = % di pazienti dentro la banda di normalita' di Les (media ±1.96 SD)")
    A("  TOST          = test di EQUIVALENZA entro ±margin (p<α ⇒ equivalente).")
    A("                  'NON equivalente' non prova che sia diverso: leggere con Welch/d.")
    A("  Welch d       = dimensione dell'effetto (|d|: 0.2 piccolo, 0.5 medio, 0.8 grande)")
    A("  corr / RMSE   = somiglianza di FORMA vs Les (dopo allineamento e normalizzazione)")

    # ---- 1. QC: chiusura di massa ----
    mc = np.array([r["mass_closure"] for r in records], float)
    bad = int(np.sum(np.abs(mc - 1) > 0.02))
    A("\n" + sub)
    A("1. CONTROLLO QUALITA' — chiusura di massa")
    A(sub)
    A(f"  media {np.nanmean(mc):.4f}  (min {np.nanmin(mc):.4f}, max {np.nanmax(mc):.4f})")
    A(f"  pazienti oltre il 2% di sbilanciamento: {bad}/{len(mc)}")
    if bad:
        A("  -> se >0, controllare lettura facce/segni PRIMA di interpretare le frazioni.")
    else:
        A("  -> conservazione di massa rispettata: le frazioni sono affidabili.")

    # ---- 2. LIVELLO 1: frazioni vs letteratura (Les) ----
    A("\n" + sub)
    A("2. LIVELLO 1 (robusto) — frazioni vs Les et al. (n=36)")
    A(sub)
    if "level1_iliac" in stats_out:
        s = stats_out["level1_iliac"]; t = s["tost"]; w = s["welch"]
        A(f"  2a) ILIACHE (L+R)  vs  IR/SC = {s['ref_mean']:.3f} ± {s['ref_sd']:.3f}")
        A(f"      coorte        : {t['mean']:.3f} ± {t['sd']:.3f}  (n={t['n']})")
        A(f"      banda Les 95% : [{s['ref_band'][0]:.3f}, {s['ref_band'][1]:.3f}]  "
          f"— coverage {s['coverage']*100:.1f}%")
        A(f"      TOST (±{cfg.tost_margin}) : p={t['p_tost']:.4f}  -> "
          f"{'EQUIVALENTE' if t['equivalent'] else 'NON equivalente'}")
        A(f"      Welch         : diff={w['diff']:+.3f}  "
          f"IC95 [{w['ci95'][0]:+.3f}, {w['ci95'][1]:+.3f}]  p={w['p']:.4f}  d={w['cohens_d']:+.2f}")
        A(f"      lettura       : {_fmt_interpretation_l1(s, cfg)}")
    if "level1_visceral_renal" in stats_out:
        s = stats_out["level1_visceral_renal"]; t = s["tost"]
        A(f"\n  2b) VISCERE+RENI   vs  1-IR/SC = {s['ref_mean']:.3f}")
        A(f"      coorte        : {t['mean']:.3f} ± {t['sd']:.3f}  (n={t['n']})")
        A(f"      coverage      : {s['coverage']*100:.1f}%   "
          f"TOST p={t['p_tost']:.4f} ({'EQUIVALENTE' if t['equivalent'] else 'NON equiv.'})")
        A("      nota          : complementare delle iliache per conservazione di massa;")
        A("                      SD di riferimento presa uguale a quella di IR/SC (approssimazione).")

    # ---- 3. LIVELLO 2: split interno vs tesi (descrittivo) ----
    A("\n" + sub)
    A("3. LIVELLO 2 (descrittivo) — split interno vs tesi Surianarayanan (1 pz, NO SD)")
    A(sub)
    A("  Solo scarto: 1 solo paziente di riferimento -> nessun test statistico lecito.")
    A(f"  {'ramo':9s} {'coorte':>16s}  {'tesi':>7s}  {'scarto':>9s}")
    for _, r in stats_out["level2_thesis"].iterrows():
        A(f"  {r['ramo']:9s} {r['coorte_mean']:.3f} ± {r['coorte_sd']:.3f}  "
          f"{r['tesi_ref']:>7.3f}  {r['scarto_abs']:+.3f} ({r['scarto_rel']*100:+.1f}%)")

    # ---- 4. FORMA vs Les (se disponibile) ----
    if shape_df is not None and not shape_df.empty:
        A("\n" + sub)
        A("4. FORMA vs Les — correlazione e RMSE (sola forma d'onda)")
        A(sub)
        A("  Piano SEPARATO dalle frazioni: qui la media e' rimossa (pulse). Forma e")
        A("  magnitudine possono concordare/discordare in modo indipendente.")
        for agg in REF_WAVES:
            s = shape_df[shape_df["aggregato"] == agg]
            if not s.empty:
                tag = "  [sanity check BC]" if agg == "inlet" else ""
                A(f"  {agg:15s}: corr {s['corr_vs_les'].mean():.3f} ± {s['corr_vs_les'].std():.3f}, "
                  f"RMSE {s['rmse_vs_les'].mean():.3f}{tag}")

    # ---- 5. Diagnostica: reverse diastolico renale ----
    A("\n" + sub)
    A(f"5. DIAGNOSTICA — reverse diastolico renale (soglia {cfg.reverse_flag_threshold})")
    A(sub)
    for rlabel in ["renal_L", "renal_R"]:
        vals = np.array([f[rlabel]["rev_time_frac"] for f in features_all.values() if rlabel in f], float)
        if vals.size:
            n_over = int(np.sum(vals > cfg.reverse_flag_threshold))
            A(f"  {rlabel}: reverse medio {np.nanmean(vals):.3f}; {n_over}/{vals.size} pz sopra soglia")
    A("  -> un reverse renale diffuso puo' essere artefatto RCR (res1_split uniforme):")
    A("     l'R1 e' il 15% fisso di R_tot per ogni outlet, non tarato per ramo. Da verificare.")

    # ---- chiave di lettura complessiva ----
    A("\n" + line)
    A("COME PRESENTARE — ordine di lettura:")
    A("  (1) QC massa OK  ->  (2) frazioni L1: quanto e dove si scosta da Les  ->")
    A("  (4) forma: lo scostamento e' di ripartizione o anche di profilo?  ->")
    A("  ricordare che con BC Murray-RCR la ripartizione media e' in gran parte")
    A("  PRESCRITTA: uno scostamento L1 e' anzitutto su geometria+disegno BC.")
    A(line)

    rep = "\n".join(L)
    with open(os.path.join(outdir, "report.txt"), "w", encoding="utf-8") as fh:
        fh.write(rep + "\n")
    try:
        print(rep)
    except UnicodeEncodeError:                      # console Windows cp1252
        print(rep.encode("ascii", "replace").decode("ascii"))


# ============================================================================
# ANALYZE + DEMO
# ============================================================================

def analyze_mode(args):
    cfg = Config(cardiac_period=args.period, tost_margin=args.tost_margin,
                 shape_norm=getattr(args, "shape_norm", "pulse"),
                 shape_align=not getattr(args, "no_align", False))
    os.makedirs(args.outdir, exist_ok=True)
    fig_dir = os.path.join(args.outdir, "figures"); os.makedirs(fig_dir, exist_ok=True)

    # 1) etichette + elenco pazienti (esclude i file label_map*)
    generic, specific = load_label_map(args.label_map)
    patients = sorted(os.path.splitext(f)[0] for f in os.listdir(args.data_root)
                      if f.endswith(".csv") and not f.startswith("label_map"))
    # 2) elabora ogni paziente; salta (con messaggio) quelli che sollevano errori
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

    # 3) aggregazione + statistica
    frac_df, terr = aggregate(records, fractions_all, cfg)
    stats_out = run_statistics(terr, cfg)

    # 4) dump tabellari (tracciabilita' completa)
    frac_df.to_csv(os.path.join(args.outdir, "frazioni_per_paziente.csv"))
    terr.to_csv(os.path.join(args.outdir, "territori_per_paziente.csv"))
    terr.agg(["mean", "std", "min", "max"]).T.to_csv(os.path.join(args.outdir, "riepilogo_territori.csv"))
    stats_out["level2_thesis"].to_csv(os.path.join(args.outdir, "livello2_vs_tesi.csv"), index=False)
    feat_rows = [{"patient": p, "outlet": lab, **fv}
                 for p, fd in features_all.items() for lab, fv in fd.items()]
    pd.DataFrame(feat_rows).to_csv(os.path.join(args.outdir, "feature_forma.csv"), index=False)

    # 5) metriche di forma vs Les
    shape_df, target = shape_metrics_vs_les(norm_all, cfg)
    shape_df.to_csv(os.path.join(args.outdir, "forma_vs_les_per_paziente.csv"), index=False)

    # 6) figure + report (ora il report include anche la sezione FORMA)
    spaghetti_per_outlet(norm_all, cfg, fig_dir, target)
    aggregate_vs_les(norm_all, cfg, fig_dir, target, shape_df)
    three_panel_compare(norm_all, records, cfg, fig_dir, target)
    fractions_summary(terr, stats_out, cfg, fig_dir)
    write_report(records, terr, stats_out, features_all, cfg, args.outdir, shape_df=shape_df)

    if not shape_df.empty:
        print(f"\n[FORMA] vs Les ({cfg.shape_norm}, "
              f"{'peak-aligned' if cfg.shape_align else 'non allineate'}):")
        for agg in REF_WAVES:
            s = shape_df[shape_df["aggregato"] == agg]
            if not s.empty:
                tag = " (sanity check BC)" if agg == "inlet" else ""
                print(f"    {agg:15s} corr {s['corr_vs_les'].mean():.3f} ± {s['corr_vs_les'].std():.3f}, "
                      f"RMSE {s['rmse_vs_les'].mean():.3f}{tag}")
    print(f"\nOutput in: {args.outdir}")


def demo_mode(args):
    """Genera CSV + label_map SINTETICI (finti) e lancia analyze. Utile per testare
    la pipeline analyze senza VTU reali. NB: i dati NON sono fisici."""
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
    pa.add_argument("--shape-norm", choices=["pulse", "mean"], default="pulse",
                    help="'pulse'=demean+ampiezza (forma pura); 'mean'=divide-by-mean")
    pa.add_argument("--no-align", action="store_true", help="disattiva il peak-alignment")

    # NB: demo_mode e' referenziato nel dispatch ma NON aveva un subparser registrato
    # nella versione originale -> era di fatto irraggiungibile da CLI. Registrato qui.
    pd_ = sub.add_parser("demo", help="dati SINTETICI + analyze (test pipeline)")
    pd_.add_argument("--outdir", default="./results_demo")
    pd_.add_argument("--tost-margin", type=float, default=0.05)

    args = ap.parse_args()
    if args.mode == "export":
        export_mode(args)
    elif args.mode == "analyze":
        analyze_mode(args)
    elif args.mode == "demo":
        demo_mode(args)


if __name__ == "__main__":
    main()