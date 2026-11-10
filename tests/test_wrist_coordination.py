from copy import deepcopy

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from real2sim2real.controller_actor_training.wrist_coordination import (
    ObjectFrameCoordinator,
    rigid_wrist_targets,
    bounded_vector,
    hold_world_motion,
)


def settings():
    return dict(
        translation_gain_s_inv=1.0,
        rotation_gain_s_inv=2.0,
        max_translation_speed_m_s=0.02,
        max_rotation_speed_rad_s=0.2,
        max_translation_m=0.025,
        max_rotation_rad=0.2,
    )


def test_common_rotation_orbits_both_wrists_and_preserves_relative_frame():
    frames = np.tile(np.eye(4), (2, 1, 1))
    frames[0, :3, 3] = [1, 0, 0]
    frames[1, :3, 3] = [0, 2, 0]
    frames[1, :3, :3] = Rotation.from_euler("x", 0.3).as_matrix()
    r = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    result = rigid_wrist_targets(frames, np.zeros(3), r, np.array([0.1, 0, 0]))
    np.testing.assert_allclose(
        result[:, :3, 3], [[0.1, 1, 0], [-1.9, 0, 0]], atol=1e-10
    )
    np.testing.assert_allclose(
        np.linalg.inv(result[0]) @ result[1],
        np.linalg.inv(frames[0]) @ frames[1],
        atol=1e-10,
    )


def test_feedback_opposes_world_drift_and_respects_speed_and_travel():
    c = ObjectFrameCoordinator.__new__(ObjectFrameCoordinator)
    c.settings = settings()
    c.target_rotation = np.eye(3)
    c.pivot = None
    c.translation = np.zeros(3)
    c.rotation = np.eye(3)
    pose = np.eye(4)
    c.update(pose, 0.1)
    pose[0, 3] = 0.1
    pose[2, 3] = -0.2
    pose[:3, :3] = Rotation.from_euler("z", 0.2).as_matrix()
    c.update(pose, 0.1)
    np.testing.assert_allclose(c.translation, [-0.002, 0, 0])
    assert Rotation.from_matrix(c.rotation).as_rotvec()[2] < 0
    for _ in range(1000):
        c.update(pose, 0.1)
    assert np.linalg.norm(c.translation) <= 0.025 + 1e-12
    assert Rotation.from_matrix(c.rotation).magnitude() <= 0.2 + 1e-12
    assert c.translation[2] == 0  # independent clearance loop owns vertical motion
    with pytest.raises(ValueError):
        bounded_vector([float("nan"), 0, 0], 0.01)


def test_world_audit_detects_motion_hidden_by_comoving_wrists():
    rows = [
        dict(phase="hold", object_position_m=[x, 0, 1], wrist_position_m=[x, 0, 0])
        for x in (0, 0.03)
    ]
    assert hold_world_motion(rows)["hold_world_horizontal_drift_m"] == pytest.approx(
        0.03
    )
    assert hold_world_motion([])["hold_world_horizontal_drift_m"] is None


def test_coordinated_fixture_commands_leave_object_and_physical_state_untouched():
    mj = pytest.importorskip("mujoco")
    from real2sim2real.controller_actor_training.object_grasp_pilot import (
        set_root_frame,
    )

    def hand(name):
        joints = "".join(
            f'<joint name="{name}_{a}" type="{kind}" axis="{axis}" range="{limits}"/>'
            for a, kind, axis, limits in [
                ("x", "slide", "1 0 0", "-1 1"),
                ("y", "slide", "0 1 0", "-1 1"),
                ("z", "slide", "0 0 1", "-1 1"),
                ("rx", "hinge", "1 0 0", "-.5 .5"),
                ("ry", "hinge", "0 1 0", "-.5 .5"),
                ("rz", "hinge", "0 0 1", "-.5 .5"),
            ]
        )
        return f'<body name="{name}">{joints}<geom type="sphere" size=".01" mass="1"/></body>'

    m = mj.MjModel.from_xml_string(
        '<mujoco><compiler angle="radian"/><worldbody>'
        + hand("a")
        + hand("b")
        + '<body name="object" pos="0 0 1"><freejoint/><geom type="sphere" size=".01" mass="1"/></body></worldbody></mujoco>'
    )
    d = mj.MjData(m)
    set_root_frame(m, d, "a", "a_", [-0.2, 0, 0.2], np.eye(3))
    set_root_frame(
        m,
        d,
        "b",
        "b_",
        [0.2, 0, 0.2],
        Rotation.from_euler("xyz", [0.4, -0.3, 0.5]).as_matrix(),
    )
    c = ObjectFrameCoordinator(m, [("a", "a_"), ("b", "b_")], settings(), np.eye(3))
    original = d.qpos.copy()
    c.target(m, original, np.eye(4), 0.002)
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_euler("z", 0.1).as_matrix()
    pose[0, 3] = 0.01
    result = c.target(m, original, pose, 0.01)
    np.testing.assert_array_equal(d.qpos, original)
    np.testing.assert_array_equal(result[12:], original[12:])
    assert not np.array_equal(result[:12], original[:12])
    assert c.max_position_residual_m < 1e-10 and c.max_rotation_residual_rad < 1e-10
    assert not c.rejection
    # One wrist starts almost at its rotational limit. A common projected
    # correction must preserve the relative frames instead of clipping one.
    limited = original.copy()
    limited[5] = 0.499
    policy = settings()
    policy["joint_limit_policy"] = "project_common"
    projected = ObjectFrameCoordinator(m, [("a", "a_"), ("b", "b_")], policy, np.eye(3))
    pose[:3, :3] = Rotation.from_euler("z", -0.3).as_matrix()
    for _ in range(20):
        commands = projected.target(m, limited, pose, 0.02)
    assert projected.projected_steps > 0
    assert not projected.rejection
    assert commands[5] <= 0.5
    assert projected.max_position_residual_m < 1e-10
    assert projected.max_rotation_residual_rad < 1e-10


def test_frozen_validation_rejects_changed_coordinated_controller():
    from real2sim2real.controller_actor_training.object_grasp_pilot import (
        verify_frozen_controller,
    )

    config = dict(object_frame_coordination=settings())
    selected = dict(controller=deepcopy(config))
    verify_frozen_controller(config, selected)
    config["object_frame_coordination"]["translation_gain_s_inv"] = 2.0
    with pytest.raises(ValueError, match="feedback"):
        verify_frozen_controller(config, selected)
