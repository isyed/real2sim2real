import numpy as np
import pytest
from real2sim2real.controller_actor_training.contact_conditioned_posture import (
    ContactConditionedPosture,
)


def controller(gain=1):
    spec = {
        "gain": gain,
        "phases": ["lift", "invert"],
        "loss_threshold_n": 0.1,
        "loss_dwell_s": 0.02,
        "offsets_rad": [[0.02, 0], [0, 0.01]],
    }
    config = {
        "finger_joints": ["a", "b"],
        "pressure_groups": [[0], [1]],
        "feedback_s": 0.005,
        "object_weight_n": 2,
        "schedule": {"pressure_start_reference_s": 1, "release_reference_s": 3},
    }
    return ContactConditionedPosture(
        spec,
        config,
        {"phases": ["lift", "invert", "release"]},
        [[-1, 1], [-1, 1]],
        {"maximum_finger_offset_rad": 0.05, "maximum_offset_speed_rad_s": 0.2},
    )


def step(c, **kw):
    args = {
        "reference_time": 2,
        "phase": "invert",
        "forces_n": [0, 1],
        "support_n": 0,
        "releasing": False,
    }
    args.update(kw)
    return c.command(np.zeros(2), np.zeros(2), **args)


def test_dwell_ownership_rate_and_recovery():
    c = controller()
    for _ in range(3):
        np.testing.assert_array_equal(step(c), [0, 0])
    np.testing.assert_allclose(step(c), [0.001, 0])
    previous = np.zeros(2)
    for _ in range(40):
        current = step(c)
        assert np.max(abs(current - previous)) <= 0.002 + 1e-12
        previous = current
    np.testing.assert_allclose(current, [0.02, 0])
    for _ in range(30):
        current = step(c, forces_n=[1, 1])
    np.testing.assert_allclose(current, [0, 0], atol=1e-12)


def test_supported_missing_and_release_cannot_trigger_seating():
    for overrides in [
        {"support_n": 2},
        {"forces_n": [float("nan"), 1]},
        {"forces_n": None, "support_n": None},
        {"phase": "release"},
        {"reference_time": 3},
        {"releasing": True},
    ]:
        c = controller()
        for _ in range(20):
            np.testing.assert_array_equal(step(c, **overrides), [0, 0])
    c = controller()
    for _ in range(30):
        step(c)
    np.testing.assert_array_equal(step(c, releasing=True), [0, 0])


def test_zero_gain_is_exact_bypass_and_combined_offsets_stay_bounded():
    c = controller(0)
    a = np.array([0.1, 0.2])
    assert (
        c.command(
            a,
            a,
            reference_time=2,
            phase="invert",
            forces_n=[0, 0],
            support_n=0,
            releasing=False,
        )
        is a
    )
    c = controller()
    previous = np.zeros(2)
    for _ in range(150):
        q = c.command(
            np.zeros(2),
            np.array([0.049, 0]),
            reference_time=2,
            phase="invert",
            forces_n=[0, 0],
            support_n=0,
            releasing=False,
        )
        assert np.max(abs(q)) <= 0.05 + 1e-12
        assert np.max(abs(q - previous)) <= 0.001 + 1e-12
        previous = q


def test_illegal_gain_rejected():
    with pytest.raises(ValueError):
        controller(float("nan"))
