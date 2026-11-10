import numpy as np
import pytest

from real2sim2real.controller_actor_training.arm_path_audit import (
    interval_speed,
    fit_pose,
)


def test_speed_respects_saved_clock_and_rejects_clock_reset():
    q = np.array([[0.0, 0.0], [0.1, -0.2], [0.3, -0.3]])
    np.testing.assert_allclose(interval_speed(q, [0, 0.1, 0.3], [2, 4]), [0.5, 0.5])
    with pytest.raises(ValueError, match="timestamps"):
        interval_speed(q, [0, 0.1, 0], [2, 4])


def test_ik_tracks_body_frame_without_changing_free_object():
    mujoco = pytest.importorskip("mujoco")
    m = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
    <body><joint name="shoulder" axis="0 0 1" range="-170 170"/>
    <geom type="capsule" fromto="0 0 0 1 0 0" size=".02"/>
    <body pos="1 0 0"><joint name="elbow" axis="0 0 1" range="-170 170"/>
    <geom type="capsule" fromto="0 0 0 1 0 0" size=".02"/>
    <body name="hand" pos="1 0 0"/></body></body>
    <body pos="0 0 2"><freejoint name="object"/><geom size=".03"/></body>
    </worldbody></mujoco>""")
    d = mujoco.MjData(m)
    d.qpos[:2] = [0.4, -0.7]
    mujoco.mj_forward(m, d)
    b = m.body("hand").id
    target = np.eye(4)
    target[:3, 3] = d.xpos[b]
    target[:3, :3] = d.xmat[b].reshape(3, 3)
    object_before = d.qpos[2:].copy()
    _, pe, re = fit_pose(m, d, b, ["shoulder", "elbow"], target, np.zeros(2), 100)
    assert pe < 1e-5 and re < 1e-5
    np.testing.assert_array_equal(d.qpos[2:], object_before)


def test_joint_neighbor_filter_does_not_hide_environment_or_folded_arm_collision():
    from types import SimpleNamespace
    from real2sim2real.controller_actor_training.arm_path_audit import joint_neighbors

    # world -> torso -> fixed mount -> shoulder -> elbow; separate object
    m = SimpleNamespace(
        body_parentid=[0, 0, 1, 2, 3, 0], body_jntnum=[0, 0, 0, 1, 1, 1]
    )
    assert joint_neighbors(m, 1, 3)
    assert joint_neighbors(m, 3, 1)
    assert not joint_neighbors(m, 1, 4)
    assert not joint_neighbors(m, 0, 3)
    assert not joint_neighbors(m, 3, 5)
