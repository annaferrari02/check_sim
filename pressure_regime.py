#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pressure_regime_check.py
========================

Diagnostica di convergenza della PRESSIONE su una singola simulazione (es. pz092
a 5 cicli) letta direttamente dai result_*.vtu, e "triple check" della resistenza
del dominio 3D.

Identificazione delle facce: dai file .vtp di  mesh-complete/mesh-surfaces/
(uno per faccia: inlet + ogni outlet + wall). Ogni cap-vtp porta 'GlobalNodeID',
che combacia con il 'GlobalNodeID' del volume result_*.vtu -> matching ESATTO
dei nodi di ogni faccia (nessun modello, nessuna soglia). Se GlobalNodeID manca,
si ricade sul matching per coordinate (i nodi del cap sono comunque un
sottoinsieme dei nodi del volume, quindi la distanza di match e' ~0).

Tre verifiche (logica invariata):
  A. REGIME       -> la media di pressione agli outlet e' periodica?
  B. OUTLET==MAP  -> a regime la media outlet deve tendere a MAP (RCR: R_tot=MAP/q).
  C. Δp 3D        -> P_inlet - P_outlet (ultimo ciclo): se ~0, resistenza 3D trascurabile.

USO
---
  python3 pressure_regime_check.py \
      --results-dir  "D:/database_sim/pz092/72-procs" \
      --surfaces-dir "D:/database_sim/pz092/mesh-complete/mesh-surfaces" \
      --dt 0.0025 --period 0.8 --map-mmhg 90 \
      --outdir ./regime_pz092

  # prima, per vedere come sono nominate le facce e quale e' l'inlet:
  #   ... --list-faces
  # l'inlet e' riconosciuto dal nome file (default: contiene 'inflow' o 'inlet');
  # i wall dal prefisso 'wall'. Se i tuoi nomi differiscono usa
  #   --inlet-names "<sottostringa>"   --wall-pattern "<prefisso>"
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

import vtk
from vtk.util.numpy_support import vtk_to_numpy

MMHG_PER_DYN = 1.0 / 1333.22            # 1 dyn/cm² = 1/1333.22 mmHg


# ----------------------------------------------------------------------------
# I/O di base
# ----------------------------------------------------------------------------

def _read_vtu(path):
    reader = (vtk.vtkXMLPUnstructuredGridReader() if path.endswith(".pvtu")
              else vtk.vtkXMLUnstructuredGridReader())
    reader.SetFileName(path); reader.Update()
    return reader.GetOutput()


def _read_vtp(path):
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(path); reader.Update()
    return reader.GetOutput()


def _named_array(data, target):
    """Nome dell'array (case-insensitive) uguale a target, o None."""
    for i in range(data.GetNumberOfArrays()):
        n = data.GetArrayName(i)
        if n and n.lower() == target.lower():
            return n
    return None


def _find_pressure_array(grid, forced=None):
    pd = grid.GetPointData()
    names = [pd.GetArrayName(i) for i in range(pd.GetNumberOfArrays())]
    if forced:
        if forced not in names:
            raise ValueError(f"array '{forced}' assente. Presenti: {names}")
        return forced
    n = _named_array(pd, "pressure")
    if n:
        return n
    for nm in names:                                   # fallback: contiene 'press'
        if nm and "press" in nm.lower():
            return nm
    raise ValueError(f"nessun array di pressione tra i point-data. Presenti: {names}")


def _step_of(path):
    m = re.findall(r"(\d+)", os.path.basename(path))
    return int(m[-1]) if m else -1


def _tri_area(pts, tris):
    a, b, c = pts[tris[:, 0]], pts[tris[:, 1]], pts[tris[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)


# ----------------------------------------------------------------------------
# Caricamento facce da mesh-surfaces + matching al volume
# ----------------------------------------------------------------------------

def _classify(name, inlet_names, wall_pattern):
    low = name.lower()
    if any(low.startswith(w) or w in low for w in wall_pattern):
        return "wall"
    if any(s in low for s in inlet_names):
        return "inlet"
    return "outlet"


def load_caps(surfaces_dir, inlet_names, wall_pattern):
    """Legge ogni cap .vtp: nome, ruolo, punti, triangoli, GlobalNodeID (se c'e')."""
    files = sorted(glob.glob(os.path.join(surfaces_dir, "*.vtp")))
    if not files:
        raise ValueError(f"nessun .vtp in {surfaces_dir}")
    caps = []
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        poly = _read_vtp(f)
        tf = vtk.vtkTriangleFilter(); tf.SetInputData(poly); tf.Update(); poly = tf.GetOutput()
        pts = vtk_to_numpy(poly.GetPoints().GetData()).astype(float)
        tris = vtk_to_numpy(poly.GetPolys().GetConnectivityArray()).astype(np.int64).reshape(-1, 3)
        gname = _named_array(poly.GetPointData(), "GlobalNodeID")
        gids = vtk_to_numpy(poly.GetPointData().GetArray(gname)).astype(np.int64) if gname else None
        caps.append(dict(name=name, role=_classify(name, inlet_names, wall_pattern),
                         pts=pts, tris=tris, gids=gids, area=_tri_area(pts, tris)))
    return caps


def build_resolver(g0):
    """Prepara il matching cap->volume, una volta (topologia fissa nel tempo).
    Preferisce GlobalNodeID (esatto); tiene anche i punti per un eventuale
    fallback su coordinate."""
    vpts = vtk_to_numpy(g0.GetPoints().GetData()).astype(float)
    gname = _named_array(g0.GetPointData(), "GlobalNodeID")
    if gname is not None:
        vg = vtk_to_numpy(g0.GetPointData().GetArray(gname)).astype(np.int64)
        order = np.argsort(vg)
        return dict(kind="gid", vg=vg, order=order, vgs=vg[order], vpts=vpts)
    return dict(kind="coord", vpts=vpts)


def resolve_local(cap, R):
    """Indici LOCALI (nel volume) dei nodi del cap + QC del match.
    gid  -> QC = numero di GlobalNodeID non trovati (deve essere 0)
    coord-> QC = distanza NN massima (deve essere ~0)."""
    if R["kind"] == "gid" and cap["gids"] is not None:
        pos = np.clip(np.searchsorted(R["vgs"], cap["gids"]), 0, len(R["vgs"]) - 1)
        local = R["order"][pos]
        n_bad = int(np.sum(R["vg"][local] != cap["gids"]))
        return local, ("gid", float(n_bad))
    if "tree" not in R:
        R["tree"] = cKDTree(R["vpts"])                 # costruito una sola volta, on demand
    dist, idx = R["tree"].query(cap["pts"])
    return idx, ("coord", float(dist.max()))


# ----------------------------------------------------------------------------
# Media pesata sull'area della pressione su un cap
# ----------------------------------------------------------------------------

def cap_area_weighted(cap, P):
    """(media pesata, Σ area*P_tri, Σ area). Gli ultimi due permettono di
    aggregare piu' outlet in un'unica media pesata sull'area."""
    nodal = P[cap["local"]]
    triP = nodal[cap["tris"]].mean(axis=1)
    num = float((cap["area"] * triP).sum()); den = float(cap["area"].sum())
    return (num / den if den > 0 else np.nan), num, den


# ----------------------------------------------------------------------------
# Media di ciclo e statistica di convergenza  (INVARIATE)
# ----------------------------------------------------------------------------

def cycle_index(t, period, t0):
    return np.floor((t - t0) / period + 1e-9).astype(int)


def per_cycle_mean(t, y, period, min_samples=3):
    t = np.asarray(t, float); y = np.asarray(y, float); t0 = t[0]
    ci = cycle_index(t, period, t0)
    cyc, means = [], []
    for k in sorted(set(ci)):
        m = ci == k
        if m.sum() < min_samples:
            continue
        tt, yy = t[m], y[m]; T = tt[-1] - tt[0]
        means.append(float(np.trapezoid(yy, tt) / T) if T > 0 else float(np.mean(yy)))
        cyc.append(int(k))
    return np.array(cyc), np.array(means)


def fit_asymptote(cyc, means):
    if len(cyc) < 3:
        return None, None, None
    from scipy.optimize import curve_fit
    k = cyc.astype(float)
    try:
        p0 = (means[-1], means[-1] - means[0], 1.0)
        (yinf, A, tau), _ = curve_fit(lambda k, yinf, A, tau: yinf - A * np.exp(-k / tau),
                                      k, means, p0=p0, maxfev=10000)
        if tau <= 0 or not np.isfinite(yinf):
            return None, None, None
        resid = abs(means[-1] - yinf); target = 0.01 * abs(yinf)
        extra = tau * np.log(resid / target) if resid > target else 0.0
        return float(yinf), float(tau), max(0.0, float(extra))
    except Exception:
        return None, None, None


# ----------------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------------

def run(args):
    files = sorted(glob.glob(os.path.join(args.results_dir, args.results_glob)), key=_step_of)
    files = [f for f in files if _step_of(f) >= 0]
    if not files:
        print(f"Nessun file '{args.results_glob}' in {args.results_dir}"); sys.exit(1)
    print(f"[info] {len(files)} file, step {_step_of(files[0])}..{_step_of(files[-1])}")

    g0 = _read_vtu(files[0])
    pname = _find_pressure_array(g0, args.pressure_array)
    R = build_resolver(g0)
    print(f"[info] pressione='{pname}' | matching per "
          f"{'GlobalNodeID' if R['kind']=='gid' else 'coordinate'}")

    inlet_names = [s.strip().lower() for s in args.inlet_names.split(",") if s.strip()]
    wall_pattern = [s.strip().lower() for s in args.wall_pattern.split(",") if s.strip()]
    caps = load_caps(args.surfaces_dir, inlet_names, wall_pattern)

    worst = 0.0
    for cap in caps:
        cap["local"], (mkind, mval) = resolve_local(cap, R)
        worst = max(worst, mval)
    print(f"[QC] match nodi cap->volume "
          f"({'gid: nodi mancanti' if R['kind']=='gid' else 'coord: dist max'}): peggiore = {worst:g}")

    inlets = [c for c in caps if c["role"] == "inlet"]
    outlets = [c for c in caps if c["role"] == "outlet"]
    walls = [c for c in caps if c["role"] == "wall"]

    if args.list_faces:
        print("\n[list-faces] ruolo | nome | n_nodi | area | centroide")
        for c in caps:
            cen = c["pts"].mean(axis=0)
            print(f"  {c['role']:6s} | {c['name']:20s} | {len(c['pts']):6d} | "
                  f"area {c['area'].sum():.4g} | ({cen[0]:.2f},{cen[1]:.2f},{cen[2]:.2f})")
        return

    if len(inlets) != 1:
        print(f"[ERR] trovati {len(inlets)} inlet {[c['name'] for c in inlets]}; "
              f"usa --inlet-names per indicarne esattamente uno (vedi --list-faces)."); sys.exit(1)
    if not outlets:
        print("[ERR] nessun outlet (controlla --wall-pattern e i nomi dei file)."); sys.exit(1)
    inlet = inlets[0]
    print(f"[info] inlet='{inlet['name']}' | outlet={[c['name'] for c in outlets]} "
          f"| wall(esclusi)={[c['name'] for c in walls]}")

    # --- serie temporale ---
    to_mmhg = _unit_factor(g0, pname, args.pressure_unit)
    t, P_in, P_out = [], [], []
    per_out = {c["name"]: [] for c in outlets}
    for f in files:
        g = _read_vtu(f)
        P = vtk_to_numpy(g.GetPointData().GetArray(pname)).astype(float) * to_mmhg
        t.append(_step_of(f) * args.dt)
        P_in.append(cap_area_weighted(inlet, P)[0])
        num = den = 0.0
        for c in outlets:
            m, nu, de = cap_area_weighted(c, P)
            per_out[c["name"]].append(m); num += nu; den += de
        P_out.append(num / den if den > 0 else np.nan)     # media pesata su TUTTI gli outlet
    t = np.array(t); P_in = np.array(P_in); P_out = np.array(P_out)

    cyc_i, mean_in = per_cycle_mean(t, P_in, args.period)
    cyc_o, mean_out = per_cycle_mean(t, P_out, args.period)
    if len(cyc_o) < 2:
        print("[ERR] meno di 2 cicli completi: impossibile valutare il regime."); sys.exit(1)
    dp = mean_in - mean_out
    MAP = args.map_mmhg

    L = _report(cyc_o, mean_in, mean_out, dp, MAP, args)
    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, "regime_report.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    try:
        print("\n".join(L))
    except UnicodeEncodeError:
        print("\n".join(L).encode("ascii", "replace").decode("ascii"))

    import csv
    with open(os.path.join(args.outdir, "per_cycle.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["cycle", "P_inlet_mmHg", "P_outlet_mmHg", "dp_mmHg"])
        for k, a, b, d in zip(cyc_o, mean_in, mean_out, dp):
            w.writerow([k, f"{a:.4f}", f"{b:.4f}", f"{d:.4f}"])
    with open(os.path.join(args.outdir, "outlets_last_cycle.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["outlet", "P_mean_last_cycle_mmHg", "minus_MAP_mmHg"])
        for c in outlets:
            _, m = per_cycle_mean(t, np.array(per_out[c["name"]]), args.period)
            val = m[-1] if len(m) else np.nan
            w.writerow([c["name"], f"{val:.4f}", f"{val - MAP:+.4f}"])

    _figure(t, P_in, P_out, cyc_o, mean_in, mean_out, MAP, args)
    print(f"\n[ok] output in {args.outdir}")


def _unit_factor(grid, pname, forced):
    if forced == "mmhg":
        return 1.0
    if forced == "dyn":
        return MMHG_PER_DYN
    P = vtk_to_numpy(grid.GetPointData().GetArray(pname)).astype(float)
    return MMHG_PER_DYN if np.median(np.abs(P)) > 1000 else 1.0    # 90 mmHg ~ 1.2e5 dyn/cm²


def _report(cyc, mean_in, mean_out, dp, MAP, args):
    L = []; A = L.append
    line = "=" * 72
    A(line); A("CHECK CONVERGENZA PRESSIONE + RESISTENZA DOMINIO 3D"); A(line)
    A(f"cicli valutati : {list(map(int, cyc))}")
    A(f"MAP target     : {MAP:.2f} mmHg   (periodo {args.period}s, dt {args.dt}s)")

    d_last = abs(mean_out[-1] - mean_out[-2])
    A("\n" + "-" * 72); A("A. REGIME (media outlet: ultimo vs penultimo ciclo)"); A("-" * 72)
    for k, mo in zip(cyc, mean_out):
        A(f"   ciclo {k}: P_outlet = {mo:8.3f} mmHg")
    A(f"   |ultimo - penultimo| = {d_last:.3f} mmHg ({d_last/MAP*100:.2f}% di MAP)")
    yinf, tau, extra = fit_asymptote(cyc, mean_out)
    if yinf is not None:
        A(f"   fit asintoto  : P_inf = {yinf:.2f} mmHg (tau {tau:.2f} cicli); "
          f"~{extra:.1f} cicli extra per <1%")
    A(f"   -> {'A REGIME' if d_last < args.regime_tol_mmhg else 'NON ancora a regime'} "
      f"(soglia {args.regime_tol_mmhg} mmHg)")

    b = mean_out[-1] - MAP
    A("\n" + "-" * 72); A("B. OUTLET vs MAP (a regime la media outlet deve tendere a MAP)"); A("-" * 72)
    A(f"   P_outlet(ultimo) = {mean_out[-1]:.3f} mmHg   diff = {b:+.3f} mmHg ({b/MAP*100:+.2f}%)")
    okB = abs(b) < args.map_tol_mmhg
    A(f"   -> {'COINCIDE con MAP' if okB else 'NON coincide'} (soglia ±{args.map_tol_mmhg} mmHg)")
    if not okB and yinf is not None and abs(yinf - MAP) < args.map_tol_mmhg:
        A(f"      nota: l'asintoto stimato ({yinf:.1f}) e' compatibile con MAP: "
          f"servono piu' cicli, non un difetto di BC.")

    A("\n" + "-" * 72); A("C. Δp DOMINIO 3D (inlet - outlet, ultimo ciclo)"); A("-" * 72)
    A(f"   P_inlet(ultimo)  = {mean_in[-1]:.3f} mmHg")
    A(f"   P_outlet(ultimo) = {mean_out[-1]:.3f} mmHg")
    A(f"   Δp = {dp[-1]:+.3f} mmHg ({dp[-1]/MAP*100:+.2f}% di MAP)   "
      f"|Δp_last-Δp_penult| = {abs(dp[-1]-dp[-2]):.3f} mmHg")
    okC = abs(dp[-1]) / MAP < args.dp_tol_frac
    A(f"   -> resistenza 3D {'TRASCURABILE' if okC else 'NON trascurabile'} "
      f"(soglia {args.dp_tol_frac*100:.0f}% di MAP)")

    A("\n" + line)
    A("TRIPLE CHECK: " + " | ".join([
        f"regime={'OK' if d_last < args.regime_tol_mmhg else 'NO'}",
        f"outlet==MAP={'OK' if okB else 'NO'}",
        f"Δp3D<<={'OK' if okC else 'NO'}"]))
    A("Nota fisica: con RCR R_tot=MAP/q, a regime P_outlet->MAP e P_inlet=P_outlet+Δp.")
    A("Δp converge PRIMA della media assoluta (inlet e outlet derivano insieme):")
    A("il check C resta valido anche se A/B non sono ancora perfettamente a regime.")
    A(line)
    return L


def _figure(t, P_in, P_out, cyc, mean_in, mean_out, MAP, args):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))
    ax1.plot(t, P_in, color="C0", lw=1.0, label="inlet (istantaneo)")
    ax1.plot(t, P_out, color="C3", lw=1.0, label="outlet (istantaneo)")
    for k in range(int(np.floor(t[0] / args.period)), int(np.ceil(t[-1] / args.period)) + 1):
        ax1.axvline(k * args.period, color="0.85", lw=0.8, zorder=0)
    ax1.axhline(MAP, color="k", lw=1.0, ls="--", label=f"MAP {MAP:.0f}")
    ax1.set_xlabel("tempo [s]"); ax1.set_ylabel("pressione [mmHg]")
    ax1.set_title("Pressione media di faccia nel tempo"); ax1.legend(fontsize=8); ax1.grid(alpha=0.25)

    ax2.plot(cyc, mean_out, "o-", color="C3", label="outlet (media/ciclo)")
    ax2.plot(cyc, mean_in, "s-", color="C0", label="inlet (media/ciclo)")
    ax2.axhline(MAP, color="k", lw=1.0, ls="--", label=f"MAP {MAP:.0f}")
    yinf, tau, _ = fit_asymptote(cyc, mean_out)
    if yinf is not None:
        ax2.axhline(yinf, color="C3", lw=0.9, ls=":", label=f"asintoto {yinf:.1f}")
    ax2.set_xlabel("ciclo"); ax2.set_ylabel("pressione media [mmHg]")
    ax2.set_title("Convergenza per ciclo"); ax2.legend(fontsize=8); ax2.grid(alpha=0.25); ax2.set_xticks(cyc)
    fig.tight_layout(); fig.savefig(os.path.join(args.outdir, "regime_pressure.png"), dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Check regime pressione + resistenza 3D dai result_*.vtu")
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--results-glob", default="result_*.vtu")
    ap.add_argument("--surfaces-dir", required=True,
                    help="cartella mesh-complete/mesh-surfaces con un .vtp per faccia")
    ap.add_argument("--inlet-names", default="inflow,inlet",
                    help="sottostringhe che identificano il file dell'inlet")
    ap.add_argument("--wall-pattern", default="wall",
                    help="prefissi/sottostringhe dei file wall da escludere")
    ap.add_argument("--dt", type=float, required=True)
    ap.add_argument("--period", type=float, required=True)
    ap.add_argument("--map-mmhg", type=float, default=90.0)
    ap.add_argument("--pressure-array", default=None)
    ap.add_argument("--pressure-unit", choices=["auto", "dyn", "mmhg"], default="auto")
    ap.add_argument("--regime-tol-mmhg", type=float, default=1.0)
    ap.add_argument("--map-tol-mmhg", type=float, default=2.0)
    ap.add_argument("--dp-tol-frac", type=float, default=0.05)
    ap.add_argument("--list-faces", action="store_true", help="elenca le facce e termina")
    ap.add_argument("--outdir", default="./regime_check")
    run(ap.parse_args())


if __name__ == "__main__":
    main()