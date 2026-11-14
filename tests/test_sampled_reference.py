import numpy as np
import pytest
from real2sim2real.controller_actor_training.sampled_reference import (
    simulation_clock,
    validate_reference,
)


def fixture():
    config = {"duration_s": 0.1, "timestep_s": 0.001, "feedback_s": 0.005}
    times = simulation_clock(**config)
    ref = {
        "feedback_times_s": times,
        "reference_times_s": times.copy(),
        "previous_reference_times_s": np.maximum(0, times - 0.005),
        "finger_targets_rad": np.zeros((len(times), 2)),
        "object_targets_world": np.tile(np.eye(4), (len(times), 1, 1)),
        "phases": np.array(["acquisition"] * len(times)),
        "arm_times_s": np.array([0.0, 0.11]),
        "arm_q_rad": np.zeros((2, 1)),
        "pre_times_s": np.array([0.0, 0.01]),
        "pre_q_rad": np.zeros((2, 3)),
        "post_times_s": np.array([0.0, 0.01]),
        "post_q_rad": np.zeros((2, 3)),
    }
    return ref, config


def test_clock_preserves_incremental_physics_timestamps():
    ref, config = fixture()
    t, expected = 0.0, []
    for step in range(101):
        if step % 5 == 0:
            expected.append(t)
        t += 0.001
    np.testing.assert_array_equal(ref["feedback_times_s"], expected)
    validate_reference(ref, config, 2, 1)
    ref["feedback_times_s"] = np.linspace(0, 0.1, 21)
    with pytest.raises(ValueError, match="clock"):
        validate_reference(ref, config, 2, 1)


@pytest.mark.parametrize(
    "values",
    [
        (0, 0.001, 0.005),
        (0.1, 0, 0.005),
        (0.1, 0.001, 0),
        (0.1, 0.0015, 0.005),
        (0.101, 0.001, 0.005),
        (float("nan"), 0.001, 0.005),
    ],
)
def test_invalid_clock_rejected(values):
    with pytest.raises(ValueError):
        simulation_clock(*values)


@pytest.mark.parametrize("field", ["commands", "qpos", "qvel"])
def test_future_rollout_arrays_cannot_be_reference_inputs(field):
    ref, config = fixture()
    ref[field] = np.zeros((21, 4))
    with pytest.raises(ValueError, match="planned-reference"):
        validate_reference(ref, config, 2, 1)


def test_missing_nonfinite_or_wrong_joint_reference_rejected():
    ref, config = fixture()
    with pytest.raises(ValueError, match="finger_targets"):
        validate_reference(ref, config, 3, 1)
    ref["finger_targets_rad"][4, 0] = np.nan
    with pytest.raises(ValueError, match="finger_targets"):
        validate_reference(ref, config, 2, 1)
