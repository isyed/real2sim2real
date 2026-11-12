"""Describe short-timescale variation in saved trajectories, without rerendering.

This measures motion, not uncommanded tracking error or an oscillation diagnosis.
Sampling limits and deliberate nonlinear trajectories must be considered.
"""

import numpy as np
from scipy.signal import savgol_filter


def position_variation(times_s, positions_m, *, trend_window_s=0.3):
    """Return metric residuals from a local cubic trend, excluding filter edges."""
    t, p = np.asarray(times_s, float), np.asarray(positions_m, float)
    if (
        t.ndim != 1
        or p.shape != (len(t), 3)
        or len(t) < 7
        or not np.isfinite(t).all()
        or not np.isfinite(p).all()
        or not np.isfinite(trend_window_s)
        or trend_window_s <= 0
    ):
        raise ValueError("Finite sampled metric positions and trend window required")
    dt = float(np.median(np.diff(t)))
    if dt <= 0 or not np.allclose(np.diff(t), dt, atol=1e-7, rtol=1e-4):
        raise ValueError("Uniform increasing sample clock required")
    window = max(5, round(trend_window_s / dt) | 1)
    if window > len(t) - 2:
        raise ValueError("Insufficient duration for the declared trend window")
    residual = p - savgol_filter(p, window, 3, axis=0)
    norm = np.linalg.norm(residual[window // 2 : -(window // 2)], axis=1)
    return {
        "sample_hz": 1 / dt,
        "nyquist_hz": 0.5 / dt,
        "trend_window_samples": window,
        "trend_window_span_s": (window - 1) * dt,
        "interior_samples": len(norm),
        "rms_position_variation_mm": float(1000 * np.sqrt(np.mean(norm**2))),
        "peak_position_variation_mm": float(1000 * np.max(norm)),
        "is_tracking_error": False,
        "proves_oscillation": False,
    }
