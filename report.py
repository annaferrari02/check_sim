"""Assemble a single Markdown report + PNG figures from the check results."""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import DYN_PER_MMHG


def _fig_path(out_dir, patient, name):
    return os.path.join(out_dir, "figs", f"{patient}_{name}.png")


def _save(fig, path, dpi):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


# ----------------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------------
def _plot_shape(sc, title, ylabel, path, dpi):
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.plot(sc["phase"], sc["last_demeaned"], label="last (demeaned)")
    ax.plot(sc["phase"], sc["prev_demeaned"], "--", label="penultimate (demeaned)")
    ax.set_xlabel("phase [s]")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    _save(fig, path, dpi)


def _plot_murray_bar(rows, path, dpi):
    names = [r["face"] for r in rows]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(max(5.0, 0.5 * len(names) + 2), 3.4))
    ax.bar(x - 0.2, [r["murray_frac"] for r in rows], 0.4, label="Murray r^3")
    ax.bar(x + 0.2, [r["sim_frac"] for r in rows], 0.4, label="simulated")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("flow fraction")
    ax.set_title("Flow split: Murray vs simulated (last cycle)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    _save(fig, path, dpi)


# ----------------------------------------------------------------------------
# per-patient markdown
# ----------------------------------------------------------------------------
def render_patient(patient, r, out_dir, cfg):
    """r = dict of the four check results. Returns markdown string, writes figs."""
    L = [f"## {patient}\n"]

    # --- check 1 -----------------------------------------------------------
    c1 = r["check1"]
    L.append("### 1. Mean pressure drop across 3D domain (last cycle)\n")
    L.append(f"- inlet mean pressure: **{c1['p_inlet_mmHg']:.2f} mmHg** "
             f"({c1['p_inlet_dyn']:.0f} dyn/cm^2)")
    L.append(f"- flow-weighted outlet mean: **{c1['p_outlet_fw_mmHg']:.2f} mmHg**")
    L.append(f"- inlet - outlet: **{c1['diff_mmHg']:.3f} mmHg** "
             f"= {c1['diff_frac_MAP']*100:.2f}% of MAP, "
             f"{c1['diff_frac_pinlet']*100:.2f}% of inlet mean")
    L.append(f"- inlet pulse pressure: {c1['pulse_pressure_inlet_mmHg']:.2f} mmHg\n")
    L.append("| outlet | mean p [mmHg] | mean Q [mL/s] |")
    L.append("|---|---|---|")
    for o in c1["outlets"]:
        L.append(f"| {o['face']} | {o['p_mmHg']:.2f} | {o['q_mL_s']:.3f} |")
    L.append("")

    # --- check 4 (compact table + fig) ------------------------------------
    c4 = r["check4"]
    fig4 = _fig_path(out_dir, patient, "murray")
    _plot_murray_bar(c4["rows"], fig4, cfg.figure_dpi)
    L.append("### 4. Flow split vs Murray's law (last cycle)\n")
    L.append(f"- RMS relative error: **{c4['rms_rel_err']*100:.1f}%**, "
             f"max abs relative error: **{c4['max_abs_rel_err']*100:.1f}%**")
    L.append(f"- total outlet flow: {c4['total_outlet_flow_mL_s']:.2f} mL/s\n")
    L.append(f"![murray]({os.path.relpath(fig4, out_dir)})\n")
    L.append("| outlet | r [cm] | Murray frac | sim frac | rel err |")
    L.append("|---|---|---|---|---|")
    for o in c4["rows"]:
        L.append(f"| {o['face']} | {o['radius_cm']:.3f} | {o['murray_frac']:.3f} "
                 f"| {o['sim_frac']:.3f} | {o['rel_err']*100:+.1f}% |")
    L.append("")

    # --- check 2 (pressure shape) -----------------------------------------
    c2 = r["check2"]
    inl = c2["inlet"]
    fig2 = _fig_path(out_dir, patient, "pressure_shape_inlet")
    _plot_shape(inl, f"{patient} inlet pressure — last vs penultimate",
                "p - <p> [dyn/cm^2]", fig2, cfg.figure_dpi)
    L.append("### 2. Pressure waveform: shape vs vertical shift\n")
    L.append(f"Inlet — shift **{inl['shift']/DYN_PER_MMHG:.3f} mmHg**, "
             f"shape RMS/PP **{inl['rms_over_PP']*100:.1f}%**, "
             f"corr **{inl['corr']:.4f}**\n")
    L.append(f"![pshape]({os.path.relpath(fig2, out_dir)})\n")
    L.append("| face | shift [mmHg] | shape RMS/PP | corr |")
    L.append("|---|---|---|---|")
    L.append(f"| inlet | {inl['shift']/DYN_PER_MMHG:.3f} | "
             f"{inl['rms_over_PP']*100:.1f}% | {inl['corr']:.4f} |")
    for name, sc in c2["outlets"].items():
        L.append(f"| {name} | {sc['shift']/DYN_PER_MMHG:.3f} | "
                 f"{sc['rms_over_PP']*100:.1f}% | {sc['corr']:.4f} |")
    L.append("")

    # --- check 3 (flow shape) ---------------------------------------------
    c3 = r["check3"]
    L.append("### 3. Outlet flow waveform: shape vs vertical shift\n")
    L.append("| outlet | shift [mL/s] | shape RMS/PP | corr |")
    L.append("|---|---|---|---|")
    for name, sc in c3.items():
        L.append(f"| {name} | {sc['shift']:.3f} | "
                 f"{sc['rms_over_PP']*100:.1f}% | {sc['corr']:.4f} |")
    # one representative figure: the outlet with the largest shape change
    worst = max(c3.items(), key=lambda kv: (kv[1]["rms_over_PP"]
                                            if np.isfinite(kv[1]["rms_over_PP"]) else -1))
    fig3 = _fig_path(out_dir, patient, f"flow_shape_{worst[0]}")
    _plot_shape(worst[1], f"{patient} {worst[0]} flow — last vs penultimate",
                "Q - <Q> [mL/s]", fig3, cfg.figure_dpi)
    L.append("")
    L.append(f"Largest shape change: **{worst[0]}**\n")
    L.append(f"![qshape]({os.path.relpath(fig3, out_dir)})\n")

    L.append("\n---\n")
    return "\n".join(L)


def render_summary(all_results, cfg):
    """One-line-per-patient overview table across all checks."""
    L = ["# CFD post-processing report\n",
         f"T = {cfg.T} s, dt = {cfg.dt} s, cycles = {cfg.n_cycles}, "
         f"MAP = {cfg.MAP_dyn_cm2:.0f} dyn/cm^2 "
         f"({cfg.MAP_dyn_cm2/DYN_PER_MMHG:.1f} mmHg)\n",
         "## Summary\n",
         "| patient | Δp inlet-outlet [mmHg] | Δp / MAP | "
         "inlet p-shift [mmHg] | inlet shape RMS/PP | Murray RMS err |",
         "|---|---|---|---|---|---|"]
    for patient, r in all_results.items():
        if r.get("error"):
            L.append(f"| {patient} | ERROR: {r['error']} | | | | |")
            continue
        c1, c2, c4 = r["check1"], r["check2"], r["check4"]
        L.append(f"| {patient} | {c1['diff_mmHg']:.3f} | "
                 f"{c1['diff_frac_MAP']*100:.2f}% | "
                 f"{c2['inlet']['shift']/DYN_PER_MMHG:.3f} | "
                 f"{c2['inlet']['rms_over_PP']*100:.1f}% | "
                 f"{c4['rms_rel_err']*100:.1f}% |")
    L.append("\n---\n")
    return "\n".join(L)
