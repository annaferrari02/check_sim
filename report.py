"""
Batch-level report: assesses THIS launch configuration across all sims.

One Markdown file, organised by check (not by patient). Each section gives a
cohort verdict, a figure, and a compact per-sim (and, for Murray, per-vessel)
table. render_cohort() works on whatever sims are present in all_results, so it
can be called incrementally.
"""

import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import DYN_PER_MMHG
import aggregate as agg


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _figpath(out_dir, name):
    return os.path.join(out_dir, "figs", f"batch_{name}.png")


def _save(fig, path, dpi, retries=3, backoff=0.2):
    # On Windows, savefig() into a freshly-touched folder can occasionally hit
    # a transient OSError (e.g. errno 22) if AV/indexing briefly holds the
    # file we just created a moment ago for a sibling figure. Retry a few
    # times with a short backoff rather than losing the whole batch report.
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.tight_layout()
    for attempt in range(1, retries + 1):
        try:
            fig.savefig(path, dpi=dpi)
            break
        except OSError:
            if attempt == retries:
                plt.close(fig)
                raise
            time.sleep(backoff * attempt)
    plt.close(fig)


def _rel(path, out_dir):
    return os.path.relpath(path, out_dir)


def _stat_line(s, unit=""):
    return (f"median {s['median']:.2f}{unit}, mean {s['mean']:.2f}{unit}, "
            f"sd {s['sd']:.2f}, range {s['min']:.2f}–{s['max']:.2f}{unit} "
            f"(n={s['n']})")


def _frac_within(vals, lo, hi):
    v = np.asarray(vals, float)
    return int(np.sum((v >= lo) & (v <= hi))), len(v)


def _frac_below(vals, thr):
    v = np.asarray(vals, float)
    return int(np.sum(np.abs(v) <= thr)), len(v)


# ----------------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------------
def _fig_convergence(scalars, cfg, path):
    names = list(scalars)
    shift = [scalars[p]["conv_shift_mmHg"] for p in names]
    shape = [scalars[p]["conv_shape_pct"] for p in names]
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ax.scatter(shift, shape)
    for p, x, y in zip(names, shift, shape):
        ax.annotate(p, (x, y), fontsize=6, xytext=(3, 3),
                    textcoords="offset points")
    ax.axhline(cfg.shape_rms_over_pp_ok * 100, ls="--", color="grey", lw=1)
    ax.set_xlabel("inlet mean drift, last−penult [mmHg]")
    ax.set_ylabel("inlet shape change RMS/PP [%]")
    ax.set_title("Convergence per sim (near 0,0 = converged)")
    ax.grid(alpha=0.3)
    _save(fig, path, cfg.figure_dpi)


def _fig_sorted_bar(names, vals, ok_line, title, ylabel, path, cfg):
    order = np.argsort(vals)
    names = [names[i] for i in order]
    vals = np.array(vals)[order]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(max(5.0, 0.45 * len(names) + 2), 3.6))
    ax.bar(x, vals)
    if ok_line is not None:
        ax.axhline(ok_line, ls="--", color="red", lw=1,
                   label=f"reference {ok_line:g}")
        ax.legend(fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3, axis="y")
    _save(fig, path, cfg.figure_dpi)


def _fig_murray_by_vessel(vessel_stats, path, cfg):
    names = [v["face"] for v in vessel_stats]
    mean = [v["mean_pct"] for v in vessel_stats]
    sd = [v["sd_pct"] for v in vessel_stats]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(max(5.0, 0.5 * len(names) + 2), 3.8))
    ax.bar(x, mean, yerr=sd, capsize=3)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Murray rel. error [%]  (sim − Murray)")
    ax.set_title("Flow-split deviation by vessel across batch (mean ± sd)")
    ax.grid(alpha=0.3, axis="y")
    _save(fig, path, cfg.figure_dpi)


def _fig_pulse_pressure(scalars, cfg, path):
    names = list(scalars)
    pp = [scalars[p]["pp_mmHg"] for p in names]
    order = np.argsort(pp)
    names = [names[i] for i in order]
    pp = np.array(pp)[order]
    x = np.arange(len(names))
    lo, hi = cfg.pulse_pressure_band_mmHg
    fig, ax = plt.subplots(figsize=(max(5.0, 0.45 * len(names) + 2), 3.6))
    ax.axhspan(lo, hi, color="green", alpha=0.10, label=f"ref band {lo:g}–{hi:g}")
    ax.bar(x, pp)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("inlet pulse pressure [mmHg]")
    ax.set_title("Inlet pulse pressure per sim")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    _save(fig, path, cfg.figure_dpi)


# ----------------------------------------------------------------------------
# the report
# ----------------------------------------------------------------------------
def render_cohort(all_results, cfg, out_dir):
    L = []
    ok = agg.ok_patients(all_results)
    failed = agg.failed_patients(all_results)
    n_ok = len(ok)

    # ---- header / configuration ------------------------------------------
    L += [
        "# Batch report — launch configuration assessment\n",
        f"Simulations analysed: **{n_ok}** "
        f"({len(failed)} failed).  "
        f"T = {cfg.T} s, dt = {cfg.dt} s, cycles = {cfg.n_cycles}, "
        f"MAP = {cfg.MAP_dyn_cm2:.0f} dyn/cm² "
        f"({cfg.MAP_dyn_cm2/DYN_PER_MMHG:.0f} mmHg).",
        "",
        "This report evaluates how the shared boundary-condition setup behaves "
        "across the batch: convergence, whether the 3D-domain resistance is "
        "negligible, how closely the realised flow split follows Murray (and on "
        "which vessels it does not), and whether the resulting inlet pressure / "
        "pulse pressure are physiologically plausible.\n",
    ]
    if n_ok == 0:
        L.append("_No successful simulations yet._\n")
        if failed:
            L.append("Failed: " + ", ".join(f"{p} ({e})" for p, e in failed.items()))
        return "\n".join(L)

    scalars = agg.per_sim_scalars(all_results)
    olo, ohi = agg.outlet_count_range(all_results)
    L.append(f"Outlets per model: {olo}" + ("" if olo == ohi else f"–{ohi}") + ".\n")
    if failed:
        L.append("**Failed sims:** " +
                 ", ".join(f"{p} ({e})" for p, e in failed.items()) + "\n")
    L.append("\n---\n")

    # ---- 1. convergence ---------------------------------------------------
    _, shift = agg.column(scalars, "conv_shift_mmHg")
    _, shape = agg.column(scalars, "conv_shape_pct")
    n_conv, n_tot = _frac_below(shape, cfg.shape_rms_over_pp_ok * 100)
    fig = _figpath(out_dir, "convergence")
    _fig_convergence(scalars, cfg, fig)
    L += [
        "## 1. Convergence to periodicity (last vs penultimate cycle)\n",
        f"Inlet pressure **shape** settled (RMS/PP < "
        f"{cfg.shape_rms_over_pp_ok*100:g}%) in **{n_conv}/{n_tot}** sims. "
        f"Shape change across batch: {_stat_line(agg.summarize(shape), '%')}. "
        f"Mean drift last−penult: {_stat_line(agg.summarize(np.abs(shift)), ' mmHg')}.",
        "",
        "With 3 cycles the pressure mean is not expected to be fully periodic; "
        "what matters is that the waveform *shape* has converged while only the "
        "mean is still drifting.\n",
        f"![conv]({_rel(fig, out_dir)})\n",
        "| sim | mean drift [mmHg] | shape RMS/PP [%] | corr |",
        "|---|---|---|---|",
    ]
    for p in scalars:
        s = scalars[p]
        L.append(f"| {p} | {s['conv_shift_mmHg']:+.2f} | "
                 f"{s['conv_shape_pct']:.1f} | {s['conv_corr']:.4f} |")
    L.append("\n---\n")

    # ---- 2. 3D domain resistance -----------------------------------------
    names_r, dpf = agg.column(scalars, "dp_frac_MAP_pct")
    n_res, _ = _frac_below(dpf, cfg.resistance_frac_MAP_ok * 100)
    fig = _figpath(out_dir, "resistance")
    _fig_sorted_bar(names_r, list(dpf), cfg.resistance_frac_MAP_ok * 100,
                    "Inlet−outlet Δp as % of MAP, per sim",
                    "|Δp| / MAP [%]", fig, cfg)
    L += [
        "## 2. 3D-domain resistance (inlet vs flow-weighted outlet pressure)\n",
        f"|Δp|/MAP below {cfg.resistance_frac_MAP_ok*100:g}% in "
        f"**{n_res}/{len(dpf)}** sims. Across batch: "
        f"{_stat_line(agg.summarize(np.abs(dpf)), '%')}.",
        "",
        "Small Δp/MAP ⇒ the 3D domain adds little resistance and the imposed "
        "outlet RCRs dominate — the assumption behind the setup.\n",
        f"![res]({_rel(fig, out_dir)})\n",
        "| sim | Δp [mmHg] | Δp/MAP [%] |",
        "|---|---|---|",
    ]
    for p in scalars:
        s = scalars[p]
        L.append(f"| {p} | {s['dp_mmHg']:+.3f} | {s['dp_frac_MAP_pct']:+.2f} |")
    L.append("\n---\n")

    # ---- 3. Murray adherence (the headline for this config) --------------
    names_m, mrms = agg.column(scalars, "murray_rms_pct")
    n_mur, _ = _frac_below(mrms, cfg.murray_rms_ok * 100)
    fig_a = _figpath(out_dir, "murray_per_sim")
    _fig_sorted_bar(names_m, list(mrms), cfg.murray_rms_ok * 100,
                    "Murray RMS relative error, per sim",
                    "RMS rel. error [%]", fig_a, cfg)
    vessel_stats = agg.murray_by_vessel(all_results)
    fig_b = _figpath(out_dir, "murray_by_vessel")
    _fig_murray_by_vessel(vessel_stats, fig_b, cfg)
    worst = vessel_stats[0] if vessel_stats else None
    L += [
        "## 3. Flow split vs Murray's law\n",
        f"Per-sim RMS relative error below {cfg.murray_rms_ok*100:g}% in "
        f"**{n_mur}/{len(mrms)}** sims; across batch "
        f"{_stat_line(agg.summarize(mrms), '%')}.",
        "",
    ]
    if worst:
        L.append(f"**Where it deviates:** the largest systematic departure is at "
                 f"**{worst['face']}** "
                 f"({worst['mean_pct']:+.1f}% ± {worst['sd_pct']:.1f} across "
                 f"{worst['n']} sims). A consistent sign here means the config "
                 f"systematically over/under-supplies that district relative to "
                 f"the imposed r³ split.\n")
    L += [
        f"![murray_sim]({_rel(fig_a, out_dir)})\n",
        f"![murray_vessel]({_rel(fig_b, out_dir)})\n",
        "Per-vessel deviation across the batch (sorted by |mean|):\n",
        "| vessel | n | mean r [cm] | Murray frac | sim frac | rel err mean±sd [%] |",
        "|---|---|---|---|---|---|",
    ]
    for v in vessel_stats:
        L.append(f"| {v['face']} | {v['n']} | {v['mean_radius_cm']:.3f} | "
                 f"{v['mean_murray_frac']:.3f} | {v['mean_sim_frac']:.3f} | "
                 f"{v['mean_pct']:+.1f} ± {v['sd_pct']:.1f} |")
    L.append("")
    L += ["Per-sim summary:\n",
          "| sim | RMS rel err [%] | max abs rel err [%] |",
          "|---|---|---|"]
    for p in scalars:
        s = scalars[p]
        L.append(f"| {p} | {s['murray_rms_pct']:.1f} | {s['murray_max_pct']:.1f} |")
    L.append("\n---\n")

    # ---- 4. pulse pressure / compliance ----------------------------------
    _, pp = agg.column(scalars, "pp_mmHg")
    lo, hi = cfg.pulse_pressure_band_mmHg
    n_pp, _ = _frac_within(pp, lo, hi)
    _, pin = agg.column(scalars, "p_inlet_mmHg")
    fig = _figpath(out_dir, "pulse_pressure")
    _fig_pulse_pressure(scalars, cfg, fig)
    L += [
        "## 4. Inlet pressure & pulse pressure (compliance plausibility)\n",
        f"Inlet pulse pressure within the {lo:g}–{hi:g} mmHg reference band in "
        f"**{n_pp}/{len(pp)}** sims; across batch "
        f"{_stat_line(agg.summarize(pp), ' mmHg')}. "
        f"Inlet mean pressure: {_stat_line(agg.summarize(pin), ' mmHg')} "
        f"(target MAP {cfg.MAP_dyn_cm2/DYN_PER_MMHG:.0f} mmHg).",
        "",
        "Pulse pressure is a model output governed by the prescribed compliance; "
        "a value far outside the band suggests total compliance is mis-scaled "
        "(too low → PP too high, too high → PP too low).\n",
        f"![pp]({_rel(fig, out_dir)})\n",
        "| sim | inlet mean [mmHg] | sys/dia [mmHg] | PP [mmHg] |",
        "|---|---|---|---|",
    ]
    for p in scalars:
        s = scalars[p]
        L.append(f"| {p} | {s['p_inlet_mmHg']:.1f} | "
                 f"{s['p_sys_mmHg']:.0f}/{s['p_dia_mmHg']:.0f} | "
                 f"{s['pp_mmHg']:.1f} |")
    L.append("\n---\n")

    L.append("_Reference bands are indicative (Les 2010 brachial; "
             "Surianarayanan 2020 compliance tuning), not hard thresholds._\n")
    return "\n".join(L)