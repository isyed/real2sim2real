import numpy as np
import pytest
from real2sim2real.controller_actor_training.joint_boundary_path import (
    JointBoundaryPath,
    return_admitted,
)


def test_unwrapped_path_endpoints_speed_and_zero_boundary_velocity():
    p = JointBoundaryPath([0, 2, 5], [[0, 6], [1, 7], [-1, 8]])
    for t, expected in [(0, [0, 6]), (2, [1, 7]), (5, [-1, 8])]:
        q, v = p.command(t)
        np.testing.assert_array_equal(q, expected)
        np.testing.assert_array_equal(v, [0, 0])
    q, v = p.command(1)
    np.testing.assert_allclose(q, [0.5, 6.5])
    np.testing.assert_allclose(v, [0.9375, 0.9375])
    measured = np.max(
        np.abs([p.command(t)[1] for t in np.linspace(0, 5, 1001)]), axis=0
    )
    np.testing.assert_allclose(measured, p.maximum_speed_rad_s())


@pytest.mark.parametrize(
    "t,q", [([0, 0], [[1], [2]]), ([1, 2], [[1], [2]]), ([0, 2], [[1], [np.nan]])]
)
def test_invalid_path(t, q):
    with pytest.raises(ValueError):
        JointBoundaryPath(t, q)


def test_return_requires_complete_release_and_supported_stationary_object():
    args = {
        "task_completed": True,
        "release_fraction": 1,
        "hand_force_n": 0,
        "support_fraction": 1,
        "object_speed_m_s": 0.001,
    }
    assert return_admitted(**args)
    for key, value in [
        ("task_completed", False),
        ("release_fraction", 0.8),
        ("hand_force_n", 0.2),
        ("support_fraction", 0.2),
        ("object_speed_m_s", 0.1),
        ("hand_force_n", np.nan),
    ]:
        assert not return_admitted(**(args | {key: value}))
