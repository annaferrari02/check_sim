"""
The four validation checks. Each function returns a plain dict of numbers/arrays.
No printing, no plotting -- report.py owns the presentation.
"""

import numpy as np

from config import DYN_PER_MMHG
from cycles import cycle_masks, cycle_mean


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _outlets(faces):
    return [f for f in faces.values() if f.kind == "outlet"]


def _inlet(faces):
    ins = [f for f in faces.values() if f.kind == "inlet"]
    if len(ins) != 1:
        raise ValueError(f"Expected exactly 1 inlet, found {len(ins)}: "
                         f"{[f.name for f in ins]}")
    return ins[0]


def _shape_compare(times, series, cfg):
    """
    Compare last vs penultimate cycle of a scalar time series.
    Separates the vertical shift (mean drift) from the change in waveform shape.
    """
    m_last, m_prev, ph_last, ph_prev = cycle_masks(times, cfg)
    y_last = series[m_last]
    y_prev = series[m_prev]

    mean_last = cycle_mean(times[m_last], y_last, cfg.T)
    mean_prev = cycle_mean(times[m_prev], y_prev, cfg.T)
    d_last = y_last - mean_last
    d_prev = y_prev - mean_prev

    # align penultimate demeaned profile onto the last cycle's phase grid
    d_prev_al = np.interp(ph_last, ph_prev, d_prev)

    PP = float(y_last.max() - y_last.min())          # pulse (peak-to-peak)
    rms = float(np.sqrt(np.mean((d_last - d_prev_al) ** 2)))
    corr = (float(np.corrcoef(d_last, d_prev_al)[0, 1])
            if np.std(d_last) > 0 and np.std(d_prev_al) > 0 else np.nan)

    return {
        "shift": float(mean_last - mean_prev),   # vertical drift of the mean
        "rms_shape_diff": rms,                   # residual after removing mean
        "rms_over_PP": rms / PP if PP > 0 else np.nan,
        "corr": corr,                            # ~1 => same shape, just shifted
        "PP": PP,
        "mean_last": float(mean_last),
        "mean_prev": float(mean_prev),
        "phase": ph_last,
        "last_demeaned": d_last,
        "prev_demeaned": d_prev_al,
    }


# ----------------------------------------------------------------------------
# CHECK 1 -- mean pressure drop across the 3D domain (inlet vs outlets)
# ----------------------------------------------------------------------------
def check1_inlet_outlet_pressure(times, P, Q, faces, cfg):
    m_last, *_ = cycle_masks(times, cfg)
    tl = times[m_last]
    inlet = _inlet(faces)
    outs = _outlets(faces)

    p_in = cycle_mean(tl, P[inlet.name][m_last], cfg.T)

    # flow-weighted mean outlet pressure (small low-flow outlets weighted less)
    num = den = 0.0
    rows = []
    for f in outs:
        p_o = cycle_mean(tl, P[f.name][m_last], cfg.T)
        q_o = abs(cycle_mean(tl, Q[f.name][m_last], cfg.T))
        num += q_o * p_o
        den += q_o
        rows.append({"face": f.name, "p_dyn": p_o,
                     "p_mmHg": p_o / DYN_PER_MMHG, "q_mL_s": q_o})
    p_out_fw = num / den
    diff = p_in - p_out_fw

    pl = P[inlet.name][m_last]
    PP = float(pl.max() - pl.min())
    return {
        "p_inlet_dyn": p_in, "p_inlet_mmHg": p_in / DYN_PER_MMHG,
        "p_outlet_fw_dyn": p_out_fw, "p_outlet_fw_mmHg": p_out_fw / DYN_PER_MMHG,
        "diff_dyn": diff, "diff_mmHg": diff / DYN_PER_MMHG,
        "diff_frac_MAP": diff / cfg.MAP_dyn_cm2,
        "diff_frac_pinlet": diff / p_in if p_in else np.nan,
        "pulse_pressure_inlet_dyn": PP,
        "pulse_pressure_inlet_mmHg": PP / DYN_PER_MMHG,
        "outlets": rows,
    }


# ----------------------------------------------------------------------------
# CHECK 2 -- pressure waveform: shape vs vertical shift (last vs penultimate)
# ----------------------------------------------------------------------------
def check2_pressure_shape(times, P, faces, cfg):
    inlet = _inlet(faces)
    res = {"inlet": _shape_compare(times, P[inlet.name], cfg)}
    res["outlets"] = {f.name: _shape_compare(times, P[f.name], cfg)
                      for f in _outlets(faces)}
    return res


# ----------------------------------------------------------------------------
# CHECK 3 -- outlet flow waveform: shape vs vertical shift
# ----------------------------------------------------------------------------
def check3_outlet_flow_shape(times, Q, faces, cfg):
    return {f.name: _shape_compare(times, Q[f.name], cfg)
            for f in _outlets(faces)}


# ----------------------------------------------------------------------------
# CHECK 4 -- realised flow split vs Murray's law (r^3)
# ----------------------------------------------------------------------------
def check4_murray(times, Q, faces, cfg):
    m_last, *_ = cycle_masks(times, cfg)
    tl = times[m_last]
    outs = _outlets(faces)

    r3 = np.array([f.radius ** 3 for f in outs])
    murray_frac = r3 / r3.sum()
    qbar = np.array([abs(cycle_mean(tl, Q[f.name][m_last], cfg.T)) for f in outs])
    sim_frac = qbar / qbar.sum()
    rel_err = (sim_frac - murray_frac) / murray_frac

    rows = [{"face": outs[i].name, "radius_cm": outs[i].radius,
             "murray_frac": float(murray_frac[i]),
             "sim_frac": float(sim_frac[i]),
             "rel_err": float(rel_err[i])} for i in range(len(outs))]
    return {
        "rows": rows,
        "rms_rel_err": float(np.sqrt(np.mean(rel_err ** 2))),
        "max_abs_rel_err": float(np.max(np.abs(rel_err))),
        "total_outlet_flow_mL_s": float(qbar.sum()),
    }
