import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from real2sim2real.controller_actor_training.grasp_retention import GraspRetention


def controller():
    return GraspRetention(
        dict(
            gain_s_inv=1.5,
            max_speed_m_s=0.004,
            max_offset_m=0.008,
            minimum_contact_n=0.1,
            maximum_reference_error_m=0.02,
        )
    )


def update(c, p, w, r=np.eye(3), **kwargs):
    return c.update(
        p,
        w,
        r,
        0.01,
        thumb_force_n=1,
        opposing_force_n=kwargs.get("force", 1),
        reference_error_m=kwargs.get("error", 0),
    )


def test_rigid_common_motion_does_not_create_retention_correction():
    c = controller()
    update(c, [0.1, 0, 0], [0, 0, 0])
    r = Rotation.from_euler("z", 0.5).as_matrix()
    t = np.array([0, 0.2, 0.1])
    p = t + r @ np.array([0.1, 0, 0])
    np.testing.assert_allclose(update(c, p, t, r), 0, atol=1e-15)


def test_direction_contact_and_reference_guards_and_bounded_travel():
    c = controller()
    update(c, [0.1, 0, 0], [0, 0, 0])
    off = update(c, [0.11, 0, 0], [0, 0, 0])
    assert 0 < off[0] <= 0.004 * 0.01
    np.testing.assert_array_equal(update(c, [0.2, 0, 0], [0, 0, 0], force=0), off)
    np.testing.assert_array_equal(update(c, [0.2, 0, 0], [0, 0, 0], error=0.03), off)
    for _ in range(1000):
        update(c, [0.11, 0, 0], [0, 0, 0])
    assert np.linalg.norm(c.offset) <= 0.008 + 1e-12 and c.saturated > 0


def test_only_selected_robot_slides_change_and_joint_limit_rejects():
    c = controller()
    c.offset = np.array([0.001, 0, 0])
    q = np.arange(12, dtype=float) * 0.01
    before = q.copy()
    result = c.apply(q, [0, 2, 4], np.tile([-0.1, 0.1], (3, 1)))
    np.testing.assert_array_equal(q, before)
    np.testing.assert_array_equal(result[5:], before[5:])
    assert result[0] == 0.001 and result[1] == before[1]
    with pytest.raises(ValueError, match="joint_limit"):
        c.apply(q, [0, 2, 4], np.zeros((3, 2)))
