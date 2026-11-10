import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from real2sim2real.controller_actor_training.guided_motion import (
    MotionReference,
    GuidedMotion,
)
from real2sim2real.controller_actor_training.guided_motion_pilot import fidelity


def reference():
    poses = np.tile(np.eye(4), (3, 1, 1))
    poses[:, 0, 3] = [1, 1.01, 1.02]
    poses[:, :3, :3] = Rotation.from_euler("z", [[0], [0.05], [0.1]]).as_matrix()
    return MotionReference([2, 2.05, 2.1], poses, [40, 41, 42])


def test_capture_time_motion_and_missing_frames():
    ref = reference()
    p, r = ref.sample(0.025)
    np.testing.assert_allclose(p, [0.005, 0, 0], atol=1e-12)
    assert Rotation.from_matrix(r).magnitude() == pytest.approx(0.025)
    np.testing.assert_allclose(ref.sample(10)[0], [0.02, 0, 0], atol=1e-12)
    with pytest.raises(ValueError):
        MotionReference(ref.times, ref.poses, [40, 41, 43])
    with pytest.raises(ValueError):
        MotionReference([0, 0.05, 1], ref.poses, ref.frame_ids)


def test_motion_feedback_preserves_baseline_error_and_fixed_anchor():
    g = GuidedMotion.__new__(GuidedMotion)
    g.reference = reference()
    g.start = None
    g.enter(10, np.eye(4))
    pose = np.eye(4)
    p, r = g.delta(10.1)
    pose[:3, 3] = p + r @ [0.003, 0, 0]
    pose[:3, :3] = r
    g.enter(11, pose)
    np.testing.assert_allclose(
        g.feedback_pose(pose, 10.1)[:3, 3], [0.003, 0, 0], atol=1e-12
    )
    np.testing.assert_allclose(
        g.feedback_pose(pose, 10.1)[:3, :3], np.eye(3), atol=1e-12
    )
    assert g.start == 10


def test_stationary_object_does_not_pass_motion_fidelity():
    ref = reference()
    rows = [
        dict(
            time_s=10 + t,
            object_position_m=[0, 0, 0],
            object_rotation=np.eye(3).tolist(),
        )
        for t in [0, 0.05, 0.1]
    ]
    score = fidelity(rows, ref, 10, np.eye(4))
    assert score["maximum_position_error_m"] == pytest.approx(0.02)
    assert score["maximum_rotation_error_deg"] == pytest.approx(np.rad2deg(0.1))


def test_guided_commands_never_write_object_state_and_reject_limits(tmp_path):
    mj = pytest.importorskip("mujoco")
    from real2sim2real.artifacts import digest
    from real2sim2real.controller_actor_training.object_grasp_pilot import (
        set_root_frame,
    )

    def hand(name):
        axes = ["1 0 0", "0 1 0", "0 0 1"] * 2
        kinds = ["slide"] * 3 + ["hinge"] * 3
        joints = "".join(
            f'<joint name="{name}_{a}" type="{kind}" axis="{axis}" range="-1 1"/>'
            for a, kind, axis in zip(["x", "y", "z", "rx", "ry", "rz"], kinds, axes)
        )
        return f'<body name="{name}">{joints}<geom size=".01" mass="1"/></body>'

    model = mj.MjModel.from_xml_string(
        '<mujoco><compiler angle="radian"/><worldbody>'
        + hand("a")
        + hand("b")
        + '<body name="object" pos="0 0 1"><freejoint/><geom size=".01" mass="1"/></body></worldbody></mujoco>'
    )
    data = mj.MjData(model)
    for name, x in [("a", -0.1), ("b", 0.1)]:
        set_root_frame(model, data, name, name + "_", [x, 0, 0.1], np.eye(3))
    ref = reference()
    path = tmp_path / "ref.npz"
    np.savez(
        path,
        timestamps_s=ref.times,
        world_from_object=ref.poses,
        frame_ids=ref.frame_ids,
    )
    settings = dict(
        path=str(path),
        sha256=digest(path),
        mode="world_increment_after_grasp",
        limits=dict(
            maximum_translation_m=0.1,
            maximum_rotation_rad=0.2,
            maximum_speed_m_s=1,
            maximum_angular_speed_rad_s=2,
        ),
    )
    g = GuidedMotion(model, [("a", "a_"), ("b", "b_")], settings)
    original = data.qpos.copy()
    g.enter(10, np.eye(4))
    target = g.target(model, original, 10.1)
    np.testing.assert_array_equal(data.qpos, original)
    np.testing.assert_array_equal(target[12:], original[12:])
    assert not np.array_equal(target[:12], original[:12])
    assert g.max_command_error_m < 1e-10
    near_limit = original.copy()
    near_limit[0] = 0.999
    g.target(model, near_limit, 10.1)
    assert g.rejection == "reference_wrist_joint_limit"
