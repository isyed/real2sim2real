import numpy as np
import pytest
from real2sim2real.controller_actor_training.pad_pressure_correction import (
    PadPressureCorrection,
)


def settings():
    return {
        "target_force_n": 1.0,
        "gain_m_per_n_s": 0.002,
        "max_normal_speed_m_s": 0.003,
        "max_joint_speed_rad_s": 0.3,
        "max_joint_offset_rad": 0.06,
        "damping_m2": 1e-6,
    }


def test_pressure_direction_bounds_and_exclusive_finger_ownership():
    c = PadPressureCorrection([[1], [2]], np.tile([-0.5, 0.5], (4, 1)), settings())
    q = np.zeros(4)
    j = np.array([[0, 0.03, 0, 0], [0, 0, -0.03, 0]])
    for _ in range(1000):
        r = c.update(q, j, [0, 0], 0.005)
    assert r[1] < 0 and r[2] > 0 and r[0] == r[3] == 0
    assert abs(r).max() <= 0.06 + 1e-12
    before = c.offset.copy()
    r = c.update(q, j, [2, 2], 0.005)
    assert abs(r[1]) < abs(before[1]) and abs(r[2]) < abs(before[2])


def test_invalid_groups_observations_and_bound_settings_rejected():
    limits = np.tile([-0.5, 0.5], (3, 1))
    with pytest.raises(ValueError):
        PadPressureCorrection([[1], [1]], limits, settings())
    with pytest.raises(ValueError):
        PadPressureCorrection([[1]], limits, dict(settings(), max_joint_offset_rad=0.2))
    c = PadPressureCorrection([[1]], limits, settings())
    before = c.offset.copy()
    with pytest.raises(ValueError):
        c.update(np.zeros(3), np.zeros((1, 3)), [float("nan")], 0.005)
    np.testing.assert_array_equal(c.offset, before)


def test_optional_tasks_keep_legacy_exact_and_reduce_tangent_error_with_owned_joints():
    limits = np.tile([-0.5, 0.5], (3, 1))
    a = PadPressureCorrection([[0, 1]], limits, settings())
    b = PadPressureCorrection([[0, 1]], limits, settings())
    q = np.zeros(3)
    j = np.array([[0.03, 0, 0]])
    for _ in range(20):
        np.testing.assert_array_equal(
            a.update(q, j, [0.5], 0.005),
            b.update(q, j, [0.5], 0.005, targets_n=[1], tangent_tasks=[None]),
        )
    c = PadPressureCorrection([[0, 1]], limits, settings())
    task = (np.array([[0, 0, 0], [0, 0.03, 100], [0, 0, 0]]), np.array([0, 0.002, 0]))
    result = c.update(q, j, [1], 0.005, targets_n=[1], tangent_tasks=[task])
    assert result[0] == pytest.approx(0) and result[1] > 0 and result[2] == 0
    before = c.offset.copy()
    with pytest.raises(ValueError):
        c.update(q, j, [1], 0.005, tangent_tasks=[(task[0], [0, 0.02, 0])])
    np.testing.assert_array_equal(c.offset, before)
