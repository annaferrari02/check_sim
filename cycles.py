"""Cycle splitting and cycle-averaged quantities. Pure numpy, no I/O."""

import numpy as np

# numpy 2.x renamed trapz -> trapezoid
_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))


def steps_per_cycle(cfg):
    return round(cfg.T / cfg.dt)


def sanity_check_timing(cfg):
    spc = cfg.T / cfg.dt
    if abs(spc - round(spc)) > 1e-9:
        raise ValueError(
            f"T/dt = {spc} is not integer; cycle splitting assumes it is. "
            "Check T and dt in config."
        )
    return int(round(spc))


def cycle_masks(times, cfg):
    """
    Boolean masks for the last and penultimate cycle, plus the phase (0..T)
    of the samples in each. A sample on a cycle boundary is included in both
    adjacent cycles (correct for trapezoidal integration of each).
    """
    T, dt, n = cfg.T, cfg.dt, cfg.n_cycles
    tol = 0.25 * dt
    t_last0, t_last1 = (n - 1) * T, n * T
    t_prev0, t_prev1 = (n - 2) * T, (n - 1) * T
    m_last = (times >= t_last0 - tol) & (times <= t_last1 + tol)
    m_prev = (times >= t_prev0 - tol) & (times <= t_prev1 + tol)
    ph_last = times[m_last] - t_last0
    ph_prev = times[m_prev] - t_prev0
    return m_last, m_prev, ph_last, ph_prev


def cycle_mean(t, y, T):
    """Trapezoidal mean over one cycle. t should span ~T; robust if slightly off."""
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    if len(t) < 2:
        return float(np.mean(y)) if len(y) else np.nan
    span = t[-1] - t[0]
    return float(_trapz(y, t) / span)
