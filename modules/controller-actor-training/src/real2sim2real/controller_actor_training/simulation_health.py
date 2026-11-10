"""Fail closed on solver warnings or automatic simulation clock resets."""

import numpy as np


def step_rejection(before_s, after_s, timestep_s, warning_counts):
    if (
        not np.isfinite([before_s, after_s, timestep_s]).all()
        or timestep_s <= 0
        or before_s < 0
        or after_s < 0
    ):
        return "invalid_simulation_clock"
    if not np.isclose(after_s - before_s, timestep_s, atol=1e-9, rtol=1e-7):
        return "simulation_clock_reset_or_invalid_step"
    if any(count for count in warning_counts.values()):
        return "solver_warning"
    return None
