import numpy as np
import pytest
from real2sim2real.controller_actor_training.sampled_motion_variation import (
    position_variation,
)


def test_smooth_motion_is_removed_but_real_sampled_variation_is_retained():
    t = np.arange(101) * 0.02
    p = np.c_[0.3 * t**2, -0.2 * t, t * 0]
    assert position_variation(t, p)["peak_position_variation_mm"] < 1e-9
    p[:, 2] += 0.001 * (-1.0) ** np.arange(len(t))
    result = position_variation(t, p)
    assert 0.8 < result["rms_position_variation_mm"] < 1.2
    assert result["nyquist_hz"] == pytest.approx(25)
    assert not result["proves_oscillation"]


def test_irregular_clock_and_too_short_windows_are_rejected():
    t = np.arange(10) * 0.02
    with pytest.raises(ValueError, match="duration"):
        position_variation(t, np.zeros((10, 3)))
    t[-1] += 0.001
    with pytest.raises(ValueError, match="Uniform"):
        position_variation(t, np.zeros((10, 3)))
