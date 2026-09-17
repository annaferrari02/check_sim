"""
Batch-level report: assesses THIS launch configuration across all sims.

One Markdown file, organised by check (not by patient). Each section gives a
cohort verdict, figure(s), and a compact per-sim / per-vessel table.
render_cohort() works on whatever sims are present, so it can be called
incrementally.
"""

import os
from io import BytesIO
import tempfile
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
    return os.path.normpath(os.path.join(out_dir, "figs", f"batch_{name}.png"))


def _save(fig, path, dpi):
    path = os.path.normpath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.tight_layout()
    # Render in memory so Pillow does not open the Windows path itself.
    image = BytesIO()
    fig.savefig(image, format="png", dpi=dpi)
    temp_fd, temp_path = tempfile.mkstemp(
        prefix=".batch_", suffix=".png", dir=os.path.dirname(path))
    try:
        with os.fdopen(temp_fd, "wb") as fh:
            fh.write(image.getvalue())
        try:
            os.replace(temp_path, path)
        except PermissionError:
            # Windows may refuse replacing a PNG open in an image viewer.
            fallback_fd, fallback = tempfile.mkstemp(
                prefix=f"{os.path.splitext(os.path.basename(path))[0]}_",
                suffix=".png", dir=os.path.dirname(path))
            os.close(fallback_fd)
            os.replace(temp_path, fallback)
            path = fallback
    except Exception:
        try:
            os.unlink(temp_path)
        except OSError:
            pass
        raise
    plt.close(fig)
    return path


def _rel(path, out_dir):
    # forward slashes so the Markdown renders on Windows too
    return os.path.relpath(path, out_dir).replace(os.sep, "/")


def _stat_line(s, unit=""):
    if not isinstance(s, dict):
        s = agg.summarize(np.asarray(s, float))
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
    return _save(fig, path, cfg.figure_dpi)


def _fig_waveform_band(grid, lm, ls, pm, ps, ylabel, title, path, cfg):
    fig, ax = plt.subplots(figsize=(5.8, 3.9))
    ax.plot(grid, lm, color="C0", lw=1.8, label="last cycle (batch mean)")
    ax.fill_between(grid, lm - ls, lm + ls, color="C0", alpha=0.20,
                    label="±sd across sims")
    ax.plot(grid, pm, color="C1", lw=1.6, ls="--",
            label="penultimate (batch mean)")
    ax.fill_between(grid, pm - ps, pm + ps, color="C1", alpha=0.15)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xlabel("phase [s]")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3)
    return _save(fig, path, cfg.figure_dpi)


def _fig_resistance_stability(scalars, cfg, path):
    names = list(scalars)
    dl = np.array([scalars[p]["dp_mmHg"] for p in names])
    dp = np.array([scalars[p]["dp_prev_mmHg"] for p in names])
    ok = np.isfinite(dp)
    dl, dp, names = dl[ok], dp[ok], [n for n, k in zip(names, ok) if k]
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    lim = [0, max(1e-6, float(np.nanmax([dl.max() if dl.size else 0,
                                         dp.max() if dp.size else 0])) * 1.15)]
    ax.plot(lim, lim, color="grey", ls="--", lw=1, label="Δp_last = Δp_penult")
    ax.scatter(dp, dl)
    for n, x, y in zip(names, dp, dl):
        ax.annotate(n, (x, y), fontsize=6, xytext=(3, 3),
                    textcoords="offset points")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Δp penultimate cycle [mmHg]")
    ax.set_ylabel("Δp last cycle [mmHg]")
    ax.set_title("Inlet−outlet Δp: last vs penultimate\n(on the line = stable across cycles)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return _save(fig, path, cfg.figure_dpi)


def _fig_murray_parity(all_results, cfg, path):
    xs, ys, faces = agg.murray_scatter_points(all_results)
    if xs.size == 0:
        return
    uniq = sorted(set(faces))
    cmap = plt.get_cmap("tab20" if len(uniq) > 10 else "tab10")
    colors = {f: cmap(i % cmap.N) for i, f in enumerate(uniq)}
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    lo = max(1e-4, min(xs.min(), ys.min()) * 0.7)
    hi = max(xs.max(), ys.max()) * 1.3
    ax.plot([lo, hi], [lo, hi], color="grey", ls="--", lw=1, label="Murray (y=x)")
    for f in uniq:
        m = [i for i, ff in enumerate(faces) if ff == f]
        ax.scatter(xs[m], ys[m], s=28, color=colors[f], label=f, alpha=0.85)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("Murray flow fraction  r³/Σr³")
    ax.set_ylabel("simulated flow fraction")
    ax.set_title("Simulated vs Murray flow split (point = outlet × sim)")
    ax.legend(fontsize=6.5, ncol=2, loc="lower right")
    ax.grid(alpha=0.3, which="both")
    return _save(fig, path, cfg.figure_dpi)


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
    return _save(fig, path, cfg.figure_dpi)


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
    return _save(fig, path, cfg.figure_dpi)


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
    return _save(fig, path, cfg.figure_dpi)


# ----------------------------------------------------------------------------
# the report
# ----------------------------------------------------------------------------
def render_cohort(all_results, cfg, out_dir):
    L = []
    ok = agg.ok_patients(all_results)
    failed = agg.failed_patients(all_results)
    n_ok = len(ok)

    L += [
        "# Batch report — launch configuration assessment\n",
        f"Simulations analysed: **{n_ok}** ({len(failed)} failed).  "
        f"T = {cfg.T} s, dt = {cfg.dt} s, cycles = {cfg.n_cycles}, "
        f"MAP = {cfg.MAP_dyn_cm2:.0f} dyn/cm² "
        f"({cfg.MAP_dyn_cm2/DYN_PER_MMHG:.0f} mmHg).",
        "",
        "Evaluates the shared boundary-condition setup across the batch: "
        "convergence of pressure and flow, 3D-domain resistance, adherence of "
        "the realised flow split to Murray (and where it deviates), and "
        "plausibility of the inlet pressure / pulse pressure.\n",
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

    # ---- 1. pressure convergence -----------------------------------------
    _, shift = agg.column(scalars, "conv_shift_mmHg")
    _, shape = agg.column(scalars, "conv_shape_pct")
    n_conv, n_tot = _frac_below(shape, cfg.shape_rms_over_pp_ok * 100)
    fig_c = _figpath(out_dir, "convergence")
    fig_c = _fig_convergence(scalars, cfg, fig_c)
    grid, last, prev, _ = agg.demeaned_bank(
        all_results, lambda r: r["check2"]["inlet"],
        scale=1.0 / DYN_PER_MMHG, T=cfg.T)
    fig_w = _figpath(out_dir, "pressure_waveform")
    lm, lsd = agg.band(last); pm, psd = agg.band(prev)
    if lm is not None:
        fig_w = _fig_waveform_band(grid, lm, lsd, pm, psd,
                           "inlet pressure − ⟨p⟩ [mmHg]",
                           "Inlet pressure waveform (demeaned): last vs penultimate",
                           fig_w, cfg)
    L += [
        "## 1. Pressure convergence (last vs penultimate cycle)\n",
        f"Inlet pressure **shape** settled (RMS/PP < "
        f"{cfg.shape_rms_over_pp_ok*100:g}%) in **{n_conv}/{n_tot}** sims. "
        f"Shape change: {_stat_line(shape, '%')}. "
        f"Mean drift |last−penult|: {_stat_line(np.abs(shift), ' mmHg')}.",
        "",
        "The demeaned waveform (cycle mean removed) isolates the *shape*: last "
        "and penultimate curves overlapping ⇒ the pressure shape is periodic "
        "even though the mean level is still drifting at 3 cycles.\n",
        f"![pwave]({_rel(fig_w, out_dir)})\n" if lm is not None else "",
        f"![conv]({_rel(fig_c, out_dir)})\n",
        "| sim | mean drift [mmHg] | shape RMS/PP [%] | corr |",
        "|---|---|---|---|",
    ]
    for p in scalars:
        s = scalars[p]
        L.append(f"| {p} | {s['conv_shift_mmHg']:+.2f} | "
                 f"{s['conv_shape_pct']:.1f} | {s['conv_corr']:.4f} |")
    L.append("\n---\n")

    # ---- 2. flow convergence (check3, previously unreported) -------------
    dom = agg.dominant_outlet(all_results)
    vshape = agg.flow_shape_by_vessel(all_results)
    fig_fw = _figpath(out_dir, "flow_waveform")
    made_fw = False
    if dom is not None:
        grid, last, prev, _ = agg.demeaned_bank(
            all_results, lambda r: r["check3"].get(dom), scale=1.0, T=cfg.T)
        lm, lsd = agg.band(last); pm, psd = agg.band(prev)
        if lm is not None:
            fig_fw = _fig_waveform_band(grid, lm, lsd, pm, psd,
                               "flow − ⟨Q⟩ [mL/s]",
                               f"Outlet flow waveform (demeaned): {dom} — last vs penult",
                               fig_fw, cfg)
            made_fw = True
    corrs = np.array([v["corr_mean"] for v in vshape], float)
    conv_all = np.nansum(corrs > 0.99)
    L += [
        "## 2. Flow convergence (outlet waveforms, last vs penultimate)\n",
        f"Across the batch, {int(conv_all)}/{len(vshape)} outlet types have mean "
        f"shape correlation > 0.99 between the last two cycles — the moving flow "
        f"field is at regime in shape even where the mean is still drifting. "
        f"(This is check3, the flow-shape diagnostic.)\n",
    ]
    if made_fw:
        L.append(f"![fwave]({_rel(fig_fw, out_dir)})  \n"
                 f"_Dominant outlet ({dom}); last and penultimate demeaned flow "
                 f"nearly coincide._\n")
    L += [
        "Per-vessel flow-shape convergence (mean across sims):\n",
        "| vessel | n | corr | shape RMS/PP [%] | mean shift [mL/s] |",
        "|---|---|---|---|---|",
    ]
    for v in vshape:
        L.append(f"| {v['face']} | {v['n']} | {v['corr_mean']:.4f} | "
                 f"{v['rms_over_pp_mean_pct']:.1f} | {v['shift_mean']:+.3f} |")
    L.append("\n---\n")

    # ---- 3. 3D domain resistance -----------------------------------------
    names_r, dpf = agg.column(scalars, "dp_frac_MAP_pct")
    n_res, _ = _frac_below(dpf, cfg.resistance_frac_MAP_ok * 100)
    _, dpl = agg.column(scalars, "dp_mmHg")
    _, dstab = agg.column(scalars, "dp_stability_mmHg")
    fig_rb = _figpath(out_dir, "resistance")
    fig_rb = _fig_sorted_bar(names_r, list(dpf), cfg.resistance_frac_MAP_ok * 100,
                    "Inlet−outlet Δp as % of MAP, per sim",
                    "|Δp| / MAP [%]", fig_rb, cfg)
    fig_rs = _figpath(out_dir, "resistance_stability")
    fig_rs = _fig_resistance_stability(scalars, cfg, fig_rs)
    L += [
        "## 3. 3D-domain resistance (inlet vs flow-weighted outlet pressure)\n",
        f"|Δp|/MAP below {cfg.resistance_frac_MAP_ok*100:g}% in "
        f"**{n_res}/{len(dpf)}** sims. Δp/MAP: {_stat_line(np.abs(dpf), '%')}.",
        "",
        f"**Absolute Δp (last cycle):** {_stat_line(np.abs(dpl), ' mmHg')}. "
        f"Since the run is not yet at regime, the absolute Δp is the more robust "
        f"reading: inlet and outlet mean pressures drift together, so their "
        f"*difference* is nearly the same on the last and penultimate cycle "
        f"(|Δp_last−Δp_penult|: {_stat_line(np.abs(dstab), ' mmHg')}). Assuming "
        f"that difference stays stable in later cycles, the last-cycle Δp is a "
        f"usable estimate of the 3D-domain impact even before convergence.\n",
        f"![resstab]({_rel(fig_rs, out_dir)})\n",
        f"![res]({_rel(fig_rb, out_dir)})\n",
        "| sim | Δp last [mmHg] | Δp penult [mmHg] | Δp/MAP [%] |",
        "|---|---|---|---|",
    ]
    for p in scalars:
        s = scalars[p]
        L.append(f"| {p} | {s['dp_mmHg']:+.3f} | {s['dp_prev_mmHg']:+.3f} | "
                 f"{s['dp_frac_MAP_pct']:+.2f} |")
    L.append("\n---\n")

    # ---- 4. Murray adherence ---------------------------------------------
    names_m, mrms = agg.column(scalars, "murray_rms_pct")
    n_mur, _ = _frac_below(mrms, cfg.murray_rms_ok * 100)
    fig_par = _figpath(out_dir, "murray_parity")
    fig_par = _fig_murray_parity(all_results, cfg, fig_par)
    vessel_stats = agg.murray_by_vessel(all_results)
    fig_ves = _figpath(out_dir, "murray_by_vessel")
    fig_ves = _fig_murray_by_vessel(vessel_stats, fig_ves, cfg)
    worst = vessel_stats[0] if vessel_stats else None
    L += [
        "## 4. Flow split vs Murray's law\n",
        f"Per-sim RMS relative error below {cfg.murray_rms_ok*100:g}% in "
        f"**{n_mur}/{len(mrms)}** sims; {_stat_line(mrms, '%')}.",
        "",
        "The parity plot shows every outlet of every sim against the Murray "
        "target: points on the diagonal follow Murray exactly, points off it are "
        "where the 3D domain redistributes flow.\n",
        f"![parity]({_rel(fig_par, out_dir)})\n",
    ]
    if worst:
        L.append(f"**Where it deviates most:** **{worst['face']}** "
                 f"({worst['mean_pct']:+.1f}% ± {worst['sd_pct']:.1f} across "
                 f"{worst['n']} sims). A consistent sign ⇒ the config "
                 f"systematically over/under-supplies that district.\n")
    L += [f"![murray_vessel]({_rel(fig_ves, out_dir)})\n",
          "Per-vessel deviation across the batch (sorted by |mean|):\n",
          "| vessel | n | mean r [cm] | Murray frac | sim frac | rel err mean±sd [%] |",
          "|---|---|---|---|---|---|"]
    for v in vessel_stats:
        L.append(f"| {v['face']} | {v['n']} | {v['mean_radius_cm']:.3f} | "
                 f"{v['mean_murray_frac']:.3f} | {v['mean_sim_frac']:.3f} | "
                 f"{v['mean_pct']:+.1f} ± {v['sd_pct']:.1f} |")
    L.append("\n---\n")

    # ---- 5. pulse pressure / compliance ----------------------------------
    _, pp = agg.column(scalars, "pp_mmHg")
    lo, hi = cfg.pulse_pressure_band_mmHg
    n_pp, _ = _frac_within(pp, lo, hi)
    _, pin = agg.column(scalars, "p_inlet_mmHg")
    fig_pp = _figpath(out_dir, "pulse_pressure")
    fig_pp = _fig_pulse_pressure(scalars, cfg, fig_pp)
    L += [
        "## 5. Inlet pressure & pulse pressure (compliance plausibility)\n",
        f"Inlet PP within {lo:g}–{hi:g} mmHg in **{n_pp}/{len(pp)}** sims; "
        f"PP {_stat_line(pp, ' mmHg')}. "
        f"Inlet mean {_stat_line(pin, ' mmHg')} "
        f"(target MAP {cfg.MAP_dyn_cm2/DYN_PER_MMHG:.0f} mmHg).",
        "",
        "PP is a model output governed by the prescribed compliance; being a "
        "within-cycle difference it is robust to the mean-pressure transient, "
        "unlike the absolute mean.\n",
        f"![pp]({_rel(fig_pp, out_dir)})\n",
        "| sim | inlet mean [mmHg] | sys/dia [mmHg] | PP [mmHg] |",
        "|---|---|---|---|",
    ]
    for p in scalars:
        s = scalars[p]
        L.append(f"| {p} | {s['p_inlet_mmHg']:.1f} | "
                 f"{s['p_sys_mmHg']:.0f}/{s['p_dia_mmHg']:.0f} | "
                 f"{s['pp_mmHg']:.1f} |")
    L.append("\n---\n")
    L.append("_Reference bands are indicative (Les 2010; Surianarayanan 2020), "
             "not hard thresholds._\n")
    return "\n".join(L)