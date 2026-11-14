import copy

import numpy as np
import pytest
from real2sim2real.controller_actor_training.reference_refinement import (
    ReferenceRefinement,
)


def inputs():
    config = {
        "pressure_targets_n": [5.0, 2.5],
        "pressure": {"gain_m_per_n_s": 0.002},
        "schedule": {"pressure_start_reference_s": 1.0, "release_reference_s": 3.0},
        "finger_joints": ["thumb", "index", "unused"],
        "pressure_groups": [[0], [1]],
        "feedback_s": 0.005,
    }
    ref = {
        "reference_times_s": np.arange(0, 4, 0.005),
        "finger_targets_rad": np.zeros((800, 3)),
    }
    bounds = {
        "maximum_knots": 12,
        "minimum_gain_scale": 0.5,
        "maximum_gain_scale": 2.0,
        "maximum_finger_offset_rad": 0.05,
        "maximum_offset_speed_rad_s": 0.2,
    }
    limits = np.tile([-1.0, 1.0], (3, 1))
    return config, ref, limits, bounds


def test_profiles_interpolate_and_restore_release_and_home():
    args = inputs()
    edit = {
        "reference_times_s": [1.0, 2.0, 3.0],
        "pressure_targets_n": [[5, 2.5], [4, 3], [5, 2.5]],
        "gain_scale": [1, 1.5, 1],
        "finger_offsets_rad": [[0, 0, 0], [0.02, -0.01, 0], [0, 0, 0]],
    }
    saved = copy.deepcopy(edit)
    profile = ReferenceRefinement(edit, *args)
    q, f, g = profile.command(1.5, np.zeros(3))
    np.testing.assert_allclose(q, [0.01, -0.005, 0])
    np.testing.assert_allclose(f, [4.5, 2.75])
    assert g == 0.0025
    for t in [0, 1, 3, 10]:
        q, f, g = profile.command(t, np.zeros(3))
        np.testing.assert_array_equal(q, np.zeros(3))
        np.testing.assert_array_equal(f, [5, 2.5])
        assert g == 0.002
    assert edit == saved


@pytest.mark.parametrize(
    "change",
    [
        {"scene": "other.xml"},
        {"reference_times_s": [1, 1, 3]},
        {"reference_times_s": [0, 2, 3]},
        {"gain_scale": [1, float("nan"), 1]},
        {"gain_scale": [1, 3, 1]},
        {"pressure_targets_n": [[5, 2.5], [5.1, 3], [5, 2.5]]},
        {"finger_offsets_rad": [[0, 0, 0], [0.051, 0, 0], [0, 0, 0]]},
        {"finger_offsets_rad": [[0, 0, 0], [0, 0, 0.01], [0, 0, 0]]},
        {"finger_offsets_rad": [[0.01, 0, 0], [0, 0, 0], [0, 0, 0]]},
    ],
)
def test_invalid_profile_rejected(change):
    edit = {"reference_times_s": [1, 2, 3]}
    edit.update(change)
    with pytest.raises(ValueError):
        ReferenceRefinement(edit, *inputs())


def test_rate_checked_on_execution_clock_and_native_limits():
    args = list(inputs())
    edit = {
        "reference_times_s": [1, 2, 3],
        "finger_offsets_rad": [[0, 0, 0], [0.04, 0, 0], [0, 0, 0]],
    }
    args[1]["reference_times_s"] = np.repeat([0.0, 2.0], 400)
    with pytest.raises(ValueError, match="rate"):
        ReferenceRefinement(edit, *args)
    args = list(inputs())
    args[2][0] = [-0.01, 0.01]
    with pytest.raises(ValueError, match="native"):
        ReferenceRefinement(edit, *args)
