import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from real2sim2real.controller_actor_training.contact_anchor import ContactAnchors


def controller():
    return ContactAnchors(
        dict(
            gain_s_inv=1, max_speed_m_s=0.001, minimum_force_n=0.5, maximum_error_m=0.01
        )
    )


def test_anchor_uses_observed_material_point_and_never_reanchors_to_hide_slip():
    c = controller()
    body, obj = np.eye(4), np.eye(4)
    point, normal = np.array([0.1, 0, 0.02]), np.array([0.0, 0, 1])
    result = c.task("pad", body, obj, point, normal, 1, active=True)
    np.testing.assert_array_equal(result[2], np.zeros(3))
    body[0, 3] = 0.004
    result = c.task("pad", body, obj, point + [0.004, 0, 0], normal, 1, active=True)
    np.testing.assert_allclose(result[2], [-0.001, 0, 0])
    np.testing.assert_array_equal(c.anchors["pad"]["object"], point)
    body[0, 3] = 0.02
    assert c.task("pad", body, obj, point, normal, 1, active=True) is None
    np.testing.assert_array_equal(c.anchors["pad"]["object"], point)


def test_anchor_normal_motion_is_uncontrolled_and_common_rigid_motion_is_invariant():
    c = controller()
    body, obj = np.eye(4), np.eye(4)
    point, normal = np.array([0.1, 0, 0.02]), np.array([0.0, 0, 1])
    c.task("pad", body, obj, point, normal, 1, active=True)
    body[2, 3] = 0.003
    np.testing.assert_allclose(
        c.task("pad", body, obj, point, normal, 1, active=True)[2], 0
    )
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler("xyz", [0.2, 0.5, -0.4]).as_matrix()
    transform[:3, 3] = [0.3, -0.1, 0.5]
    moved = c.task(
        "pad",
        transform @ body,
        transform @ obj,
        transform[:3, :3] @ point + transform[:3, 3],
        transform[:3, :3] @ normal,
        1,
        active=True,
    )
    np.testing.assert_allclose(moved[2], 0, atol=1e-15)


def test_missing_contact_and_inactive_phase_do_not_latch_or_move_anchors():
    c = controller()
    args = ("pad", np.eye(4), np.eye(4), None, [0, 0, 1], 0)
    assert c.task(*args, active=True) is None
    assert c.anchors == {}
    assert (
        c.task("pad", np.eye(4), np.eye(4), [0, 0, 0], [0, 0, 1], 1, active=False)
        is None
    )
    assert c.anchors == {}
    with pytest.raises(ValueError):
        c.task("pad", np.eye(4), np.eye(4), [0, 0, 0], [0, 0, 2], 1, active=True)
