from copy import deepcopy
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import trimesh

from real2sim2real.controller_actor_training.multi_hand_scene import namespace_robot
from real2sim2real.controller_actor_training.support_planning import (
    underside_targets,
    copy_named_state,
)
from real2sim2real.controller_actor_training.support_control import (
    contact_wrench_on_body,
    support_hold_metrics,
    support_rejections,
    SupportMotion,
)


def test_namespace_preserves_robot_references_and_scopes_defaults():
    root = ET.fromstring(
        '<mujoco><default><geom friction=".8"/><default class="finger"><joint damping="1"/></default></default><asset><mesh name="pad" file="pad.stl"/></asset><worldbody><body name="hand"><joint name="bend" class="finger"/><geom name="tip" mesh="pad"/></body></worldbody><actuator><position name="drive" joint="bend"/></actuator></mujoco>'
    )
    namespace_robot(root, "second__")
    assert root.find("worldbody/body").get("childclass") == "second__defaults"
    assert root.find(".//joint").get("damping") == "1"
    assert root.find("worldbody/body/joint").get("class") == "second__finger"
    assert root.find("actuator/position").get("joint") == "second__bend"
    assert root.find("asset/mesh").get("file") == "pad.stl"
    assert root.find("worldbody/body/geom").get("mesh") == "second__pad"
    with pytest.raises(ValueError):
        namespace_robot(
            ET.fromstring('<mujoco><include file="else.xml"/></mujoco>'), "second__"
        )


def test_underside_targets_use_bottom_not_top_or_cavity():
    mesh = trimesh.creation.box(extents=[0.2, 0.2, 0.05])
    mesh.apply_translation([0, 0, 0.1])
    targets = underside_targets(
        mesh, [0, 0], np.array([0, 1, 0]), [-0.03, -0.01, 0.01, 0.03]
    )
    np.testing.assert_allclose(targets[:, 2], 0.075)
    assert underside_targets(mesh, [0.5, 0], np.array([0, 1, 0]), [-0.03, 0.03]) is None


@pytest.mark.parametrize("object_first", [False, True])
def test_contact_force_sign_measures_upward_support_on_object(object_first):
    mj = pytest.importorskip("mujoco")
    floor = '<body name="hand"><geom name="pad" type="box" size="1 1 .02" pos="0 0 -.02"/></body>'
    obj = '<body name="object" pos="0 0 .06"><freejoint name="obj"/><geom name="ball" type="sphere" size=".05" mass=".5"/></body>'
    m = mj.MjModel.from_xml_string(
        '<mujoco><option timestep=".002"/><worldbody>'
        + (obj + floor if object_first else floor + obj)
        + "</worldbody></mujoco>"
    )
    d = mj.MjData(m)
    for _ in range(500):
        mj.mj_step(m, d)
    f = sum(
        (
            contact_wrench_on_body(m, d, i, m.body("object").id)[0]
            for i in range(d.ncon)
        ),
        np.zeros(3),
    )
    reaction = sum(
        (contact_wrench_on_body(m, d, i, m.body("hand").id)[0] for i in range(d.ncon)),
        np.zeros(3),
    )
    np.testing.assert_allclose(f, [0, 0, 0.5 * 9.81], atol=0.02)
    np.testing.assert_allclose(reaction, -f, atol=1e-10)


def test_added_hand_state_copy_preserves_other_coordinates_and_fixture_axes():
    mj = pytest.importorskip("mujoco")
    from real2sim2real.controller_actor_training.object_grasp_pilot import (
        set_root_frame,
    )

    def hand(name, prefix):
        return f'<body name="{name}"><joint name="{prefix}x" type="slide" axis="1 0 0"/><joint name="{prefix}y" type="slide" axis="0 1 0"/><joint name="{prefix}z" type="slide" axis="0 0 1"/><geom type="sphere" size=".02" mass="1"/></body>'

    original = hand("first", "fixture_")
    m0 = mj.MjModel.from_xml_string(
        "<mujoco><worldbody>" + original + "</worldbody></mujoco>"
    )
    m = mj.MjModel.from_xml_string(
        "<mujoco><worldbody>"
        + hand("second", "support_")
        + original
        + "</worldbody></mujoco>"
    )
    d = mj.MjData(m)
    d.qpos[:] = 0.4
    copy_named_state(m0, np.array([0.1, 0.2, 0.3]), m, d)
    np.testing.assert_allclose(d.qpos, [0.4, 0.4, 0.4, 0.1, 0.2, 0.3])
    old = m.jnt_axis.copy()
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    set_root_frame(m, d, "second", "support_", [1, 0, 0], rotation)
    np.testing.assert_array_equal(m.jnt_axis[3:], old[3:])
    np.testing.assert_allclose(m.jnt_axis[:3], rotation)


def test_load_sharing_requires_signed_upward_force_not_touching():
    rows = [
        dict(
            phase="hold",
            object_position_m=[0, 0, 0.1],
            support_wrist_position_m=[0, 0, 0],
            support_wrist_rotation=np.eye(3).tolist(),
            support_object_force_world_n=[10, 0, 0],
        )
        for _ in range(10)
    ]
    metric = support_hold_metrics(rows, 4.905)
    assert "insufficient_second_hand_load_sharing" in support_rejections(metric)
    for row in rows:
        row["support_object_force_world_n"] = [0, 0, 1]
    metric = support_hold_metrics(rows, 4.905)
    assert not support_rejections(metric)
    rows[-1]["object_position_m"][0] = 0.02
    assert "support_wrist_drift" in support_rejections(
        support_hold_metrics(rows, 4.905)
    )
    assert support_rejections(support_hold_metrics([], 4.905))


def test_support_motion_parks_inserts_and_does_not_chase_falling_object():
    motion = SupportMotion.__new__(SupportMotion)
    motion.plan = dict(
        q_pregrasp=[-0.1, 0, 0],
        q_open=[0, 0, 0],
        q_grasp=[0, 0, 0.004],
        reference_world_from_object=np.eye(4).tolist(),
    )
    motion.settings = dict(approach_s=1.0, seat_s=0.5, max_alignment_m=0.04)
    motion.enabled = True
    motion.start = 3.0
    motion.duration = 1.5
    motion.xyz = [0, 1, 2]
    motion.limits = np.array([[-1, 1]] * 3)
    motion.alignment = None
    motion.anchor_z = None
    motion.rejection = None
    pose = np.eye(4)
    pose[0, 3] = 0.01
    np.testing.assert_allclose(motion.target(2.0, pose, 0.07), [-0.1, 0, 0])
    np.testing.assert_allclose(motion.target(3.0, pose, 0.07), [-0.1, 0, 0])
    np.testing.assert_allclose(motion.target(4.5, pose, 0.07), [0.01, 0, 0.004])
    pose[2, 3] = -0.2
    np.testing.assert_allclose(motion.target(5.0, pose, 0.08), [0.01, 0, 0.014])
    bad = deepcopy(motion)
    bad.alignment = None
    bad.rejection = None
    np.testing.assert_allclose(bad.target(3.0, pose, 0.07), [-0.1, 0, 0])
    assert bad.rejection == "support_alignment_out_of_bounds"


def test_robot_bias_compensation_preserves_force_limits_and_object_freedom():
    mj = pytest.importorskip("mujoco")
    from real2sim2real.controller_actor_training.support_control import (
        bias_compensated_target,
    )
    from real2sim2real.retargeting.grasp_kinematics import robot_indices

    m = mj.MjModel.from_xml_string(
        '<mujoco><option timestep=".002"/><worldbody><body name="hand"><joint name="z" type="slide" axis="0 0 1" range="-1 1"/><geom type="sphere" size=".02" mass="1"/></body><body name="object" pos="1 0 2"><freejoint/><geom type="sphere" size=".02" mass="1"/></body></worldbody><actuator><position joint="z" kp="2000" kv="90" ctrlrange="-1 1" forcerange="-20 20"/></actuator></mujoco>'
    )
    d = mj.MjData(m)
    idx = robot_indices(m, "hand")
    for _ in range(500):
        mj.mj_forward(m, d)
        target = d.qpos.copy()
        target[0] = 0.1
        compensated = bias_compensated_target(m, d, target, idx)
        np.testing.assert_array_equal(compensated[1:], target[1:])
        d.ctrl[0] = compensated[0]
        mj.mj_step(m, d)
        assert abs(d.actuator_force[0]) <= 20.000001
        assert not np.any(d.xfrc_applied) and not np.any(d.qfrc_applied)
    assert d.qpos[0] == pytest.approx(0.1, abs=1e-4)
    assert d.qpos[3] < 2.0  # the unactuated object still falls under gravity


def test_support_load_feedback_direction_and_bounds():
    from real2sim2real.controller_actor_training.support_control import (
        support_load_offset,
    )

    settings = dict(
        target_weight_fraction=0.3,
        gain_m_per_n_s=0.003,
        max_speed_m_s=0.01,
        max_offset_m=0.015,
    )
    assert support_load_offset(0, 0, 5, 0.1, settings) == pytest.approx(0.00045)
    assert support_load_offset(0.015, 0, 50, 0.1, settings) == 0.015
    assert support_load_offset(0.001, 10, 5, 0.1, settings) == 0
    with pytest.raises(ValueError):
        support_load_offset(0, float("nan"), 5, 0.1, settings)


def test_frozen_support_controller_cannot_change_target_or_enabled_state():
    from real2sim2real.controller_actor_training.object_grasp_pilot import (
        verify_frozen_controller,
    )

    c = dict(
        support_controller=dict(
            enabled=True, load_feedback=dict(target_weight_fraction=0.3)
        )
    )
    selected = dict(controller={}, support_controller=deepcopy(c["support_controller"]))
    verify_frozen_controller(c, selected)
    c["support_controller"]["enabled"] = False
    with pytest.raises(ValueError, match="second-hand"):
        verify_frozen_controller(c, selected)
    assert support_rejections(
        dict(support_load_fraction=float("nan"), support_wrist_drift_m=0.0)
    ) == ["invalid_second_hand_evidence"]
