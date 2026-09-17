"""
Cohort-level aggregation across the batch of simulations.

Takes `all_results` (dict: patient -> {check1..4} or {"error": ...}) and turns
it into batch-level structures: one value per sim for each metric, plus the
Murray deviation grouped BY VESSEL across sims (so systematic under/over-supply
to a district shows up). No plotting or I/O here.
"""

import numpy as np


def ok_patients(all_results):
    """Names of sims that ran without error, in listing order."""
    return [p for p, r in all_results.items() if not r.get("error")]


def failed_patients(all_results):
    return {p: r["error"] for p, r in all_results.items() if r.get("error")}


def per_sim_scalars(all_results):
    """
    One row of headline numbers per successful sim.
    Returns dict: patient -> {metric: value}.
    """
    out = {}
    for p in ok_patients(all_results):
        r = all_results[p]
        c1, c2, c4 = r["check1"], r["check2"], r["check4"]
        inl = c2["inlet"]
        out[p] = {
            # convergence
            "conv_shift_mmHg": inl["shift"] / 1333.22,
            "conv_shape_pct": inl["rms_over_PP"] * 100.0,
            "conv_corr": inl["corr"],
            # 3D resistance
            "dp_mmHg": c1["diff_mmHg"],
            "dp_frac_MAP_pct": c1["diff_frac_MAP"] * 100.0,
            "dp_prev_mmHg": c1.get("diff_prev_mmHg", float("nan")),
            "dp_stability_mmHg": c1.get("diff_cycle_stability_mmHg", float("nan")),
            # Murray
            "murray_rms_pct": c4["rms_rel_err"] * 100.0,
            "murray_max_pct": c4["max_abs_rel_err"] * 100.0,
            # pressure / pulse pressure
            "p_inlet_mmHg": c1["p_inlet_mmHg"],
            "p_sys_mmHg": c1.get("p_inlet_sys_mmHg", np.nan),
            "p_dia_mmHg": c1.get("p_inlet_dia_mmHg", np.nan),
            "pp_mmHg": c1["pulse_pressure_inlet_mmHg"],
            "total_outlet_flow_mL_s": c4["total_outlet_flow_mL_s"],
        }
    return out


def column(scalars, key):
    """Array of one metric across sims (NaNs dropped), aligned name list."""
    names = [p for p in scalars if np.isfinite(scalars[p].get(key, np.nan))]
    vals = np.array([scalars[p][key] for p in names], float)
    return names, vals


def summarize(vals):
    """median / mean / sd / min / max of a metric across the batch."""
    if len(vals) == 0:
        return dict(n=0, median=np.nan, mean=np.nan, sd=np.nan,
                    min=np.nan, max=np.nan)
    return dict(n=len(vals), median=float(np.median(vals)),
                mean=float(np.mean(vals)), sd=float(np.std(vals, ddof=0)),
                min=float(np.min(vals)), max=float(np.max(vals)))


def murray_by_vessel(all_results):
    """
    Group the Murray relative error by outlet name across all sims.

    Returns a list of dicts (one per vessel that appears in >=1 sim), sorted by
    |mean rel err| descending — i.e. the vessels the configuration deviates on
    most come first. Each: {face, n, mean_pct, sd_pct, mean_radius_cm,
    mean_murray_frac, mean_sim_frac, values_pct}.
    """
    per_vessel = {}
    for p in ok_patients(all_results):
        for row in all_results[p]["check4"]["rows"]:
            d = per_vessel.setdefault(row["face"],
                                      {"rel": [], "r": [], "mf": [], "sf": []})
            d["rel"].append(row["rel_err"] * 100.0)
            d["r"].append(row["radius_cm"])
            d["mf"].append(row["murray_frac"])
            d["sf"].append(row["sim_frac"])
    out = []
    for face, d in per_vessel.items():
        rel = np.array(d["rel"], float)
        out.append({
            "face": face,
            "n": len(rel),
            "mean_pct": float(np.mean(rel)),
            "sd_pct": float(np.std(rel, ddof=0)),
            "mean_radius_cm": float(np.mean(d["r"])),
            "mean_murray_frac": float(np.mean(d["mf"])),
            "mean_sim_frac": float(np.mean(d["sf"])),
            "values_pct": rel,
        })
    out.sort(key=lambda x: -abs(x["mean_pct"]))
    return out


def outlet_count_range(all_results):
    counts = [len(all_results[p]["check4"]["rows"]) for p in ok_patients(all_results)]
    if not counts:
        return (0, 0)
    return (min(counts), max(counts))


# ----------------------------------------------------------------------------
# waveform banks (demeaned last/penult across sims on a common phase grid)
# ----------------------------------------------------------------------------
def _resample(phase, y, grid):
    return np.interp(grid, phase, y)


def demeaned_bank(all_results, get_shape, scale=1.0, n_grid=120, T=0.8):
    """
    Stack each sim's demeaned last/penult waveform onto a common phase grid.

    get_shape(patient_results) -> a shape dict with keys 'phase',
    'last_demeaned', 'prev_demeaned' (or None to skip that sim).
    `scale` multiplies the values (e.g. 1/DYN_PER_MMHG for mmHg).

    Returns grid, last (n×n_grid), prev (n×n_grid), names. Empty arrays if none.
    """
    grid = np.linspace(0.0, T, n_grid)
    last, prev, names = [], [], []
    for p in ok_patients(all_results):
        sc = get_shape(all_results[p])
        if sc is None:
            continue
        ph = np.asarray(sc["phase"], float)
        if ph.size < 2:
            continue
        last.append(_resample(ph, np.asarray(sc["last_demeaned"], float) * scale, grid))
        prev.append(_resample(ph, np.asarray(sc["prev_demeaned"], float) * scale, grid))
        names.append(p)
    if not names:
        return grid, np.empty((0, n_grid)), np.empty((0, n_grid)), []
    return grid, np.vstack(last), np.vstack(prev), names


def band(stack):
    """mean and std across sims at each phase point (axis 0)."""
    if stack.shape[0] == 0:
        return None, None
    return stack.mean(axis=0), stack.std(axis=0, ddof=0)


def dominant_outlet(all_results):
    """Outlet name with the largest mean simulated flow fraction across sims."""
    acc = {}
    for p in ok_patients(all_results):
        for row in all_results[p]["check4"]["rows"]:
            acc.setdefault(row["face"], []).append(row["sim_frac"])
    if not acc:
        return None
    return max(acc, key=lambda k: np.mean(acc[k]))


def flow_shape_by_vessel(all_results):
    """
    Aggregate check3 (outlet flow shape) by vessel across sims.
    Returns list of dicts sorted by mean corr ascending (worst-converged first):
    {face, n, corr_mean, rms_over_pp_mean_pct, shift_mean}.
    """
    per = {}
    for p in ok_patients(all_results):
        for face, sc in all_results[p]["check3"].items():
            d = per.setdefault(face, {"corr": [], "rms": [], "shift": []})
            d["corr"].append(sc["corr"])
            d["rms"].append(sc["rms_over_PP"] * 100.0)
            d["shift"].append(sc["shift"])
    out = []
    for face, d in per.items():
        out.append({
            "face": face, "n": len(d["corr"]),
            "corr_mean": float(np.nanmean(d["corr"])),
            "rms_over_pp_mean_pct": float(np.nanmean(d["rms"])),
            "shift_mean": float(np.nanmean(d["shift"])),
        })
    out.sort(key=lambda x: (x["corr_mean"] if np.isfinite(x["corr_mean"]) else -1))
    return out


def murray_scatter_points(all_results):
    """All (murray_frac, sim_frac, vessel) triples across sims for the parity plot."""
    xs, ys, faces = [], [], []
    for p in ok_patients(all_results):
        for row in all_results[p]["check4"]["rows"]:
            xs.append(row["murray_frac"])
            ys.append(row["sim_frac"])
            faces.append(row["face"])
    return np.array(xs), np.array(ys), faces