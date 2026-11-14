import numpy as np
import pytest
from real2sim2real.retargeting.articulated_acquisition import opposed_taper_targets


def test_insertion_template_preserves_explicit_unused_posture_and_inputs():
    from real2sim2real.retargeting.articulated_acquisition import insertion_waypoints

    pre = np.array([0.0, 0, 0, 0.2, 0.7])
    op = np.array([0.0, 0, 0, 0.4, 0.7])
    cl = np.array([0.0, 0, 0, 0.8, 0.7])
    q = insertion_waypoints(
        pre, op, cl, [0, 1, 2], [0.08, 0, 0], shape_during_insert=True
    )
    assert np.allclose(q[:, 4], 0.7)
    assert np.allclose(q[1], [0.04, 0, 0, 0.3, 0.7])
    assert pre[0] == 0 and np.array_equal(q[-1], cl)


def test_contact_chirality_mirrors_order_without_relabeling_fingers():
    p, n = opposed_taper_targets(
        [0, 0, 0], [0.035, 0.043, 0.173], 0.35, 0, 0.3, 0.01, 1
    )
    lp, ln = opposed_taper_targets(
        [0, 0, 0], [0.035, 0.043, 0.173], 0.35, 0, 0.3, 0.01, -1
    )
    assert np.allclose(lp, p * [1, -1, 1])
    assert np.allclose(ln, n * [1, -1, 1])
    assert np.allclose(np.linalg.norm(n, axis=1), 1)
    assert p[1, 1] > 0 and lp[1, 1] < 0


def test_articulated_path_rejects_collision_between_clear_endpoints():
    mj = pytest.importorskip("mujoco")
    from real2sim2real.retargeting.articulated_acquisition import (
        check_insertion_candidates,
    )
    from real2sim2real.retargeting.grasp_kinematics import robot_indices

    model = mj.MjModel.from_xml_string(
        """<mujoco><worldbody><geom name="rack" type="box" size=".005 .05 .05"/><body name="hand"><joint name="x" type="slide" axis="1 0 0" range="-1 1"/><geom name="palm" type="sphere" size=".005"/></body></worldbody></mujoco>"""
    )
    data = mj.MjData(model)
    data.qpos[0] = 0.1
    rows = check_insertion_candidates(
        model, data, robot_indices(model, "hand"), {"crossing": [[0.1], [-0.1]]}
    )
    assert not rows[0]["passed"]
    assert rows[0]["audit"]["categories"]["rack"]["penetration_m"] > 0.003
    assert data.qpos[0] == 0.1


def test_layout_rejects_contact_above_cup():
    with pytest.raises(ValueError):
        opposed_taper_targets([0, 0, 0], [0.035, 0.043, 0.173], 0.9, 0, 0.3, 0.1, 1)


def test_layout_roll_remains_on_frozen_surface_and_mirrors():
    dims = np.array([0.035, 0.043, 0.173])
    p, n = opposed_taper_targets(
        [0, 0, 0], dims, 0.35, 0, 0.3, -0.01, 1, layout_roll_rad=np.pi / 2
    )
    left, ln = opposed_taper_targets(
        [0, 0, 0], dims, 0.35, 0, 0.3, -0.01, -1, layout_roll_rad=np.pi / 2
    )
    radius = dims[0] + (dims[1] - dims[0]) * p[:, 2] / dims[2]
    assert np.allclose(np.linalg.norm(p[:, :2], axis=1), radius)
    assert p[1, 2] < p[2, 2] and np.isclose(p[1, 1], p[2, 1])
    assert np.allclose(left, p * [1, -1, 1]) and np.allclose(ln, n * [1, -1, 1])
