from types import SimpleNamespace

import numpy as np
import pytest

from real2sim2real.controller_actor_training.arm_finger_feedback import (
    ArmFingerFeedback,
    finger_waypoint,
    feedback_settings,
)


def test_nominal_finger_schedule_preserves_waypoints_and_native_clock():
    pre, opened, closed = (
        np.array([0.2, -0.1]),
        np.array([0.0, 0]),
        np.array([0.1, 0.3]),
    )
    originals = [x.copy() for x in (pre, opened, closed)]
    rollout = dict(approach_s=1.0, close_s=0.8)
    np.testing.assert_allclose(finger_waypoint(0, pre, opened, closed, rollout), pre)
    np.testing.assert_allclose(finger_waypoint(1, pre, opened, closed, rollout), opened)
    np.testing.assert_allclose(
        finger_waypoint(1.4, pre, opened, closed, rollout), closed * 0.5
    )
    np.testing.assert_allclose(finger_waypoint(5, pre, opened, closed, rollout), closed)
    for before, after in zip(originals, (pre, opened, closed)):
        np.testing.assert_array_equal(before, after)
    with pytest.raises(ValueError):
        finger_waypoint(-1, pre, opened, closed, rollout)


def test_live_binding_cannot_override_force_budgets_or_mutate_source_settings():
    from test_contact_force_control import settings

    original = settings()
    updated = feedback_settings(original, dict(normal_observation="solved_contact"))
    assert updated["normal_observation"] == "solved_contact"
    assert "normal_observation" not in original
    updated["target_force_n"]["thumb"] = 10
    assert original["target_force_n"]["thumb"] == 3
    with pytest.raises(ValueError, match="overrides"):
        feedback_settings(original, dict(max_joint_offset_rad=0.3))
    with pytest.raises(ValueError):
        feedback_settings(original, dict(normal_observation="unknown"))


def test_live_command_rebuilds_nominal_fingers_and_only_writes_owned_actuators():
    c = ArmFingerFeedback.__new__(ArmFingerFeedback)
    c.rollout = dict(approach_s=1.0, close_s=0.8)
    c.settings = dict(close_activation_fraction=0.75)
    c.hands = [
        dict(
            qids=np.array([1, 3]),
            waypoints=[np.array([0.2, 0.2]), np.zeros(2), np.array([0.1, 0.3])],
        )
    ]
    c.qids = np.array([3, 1])
    c.actuators = np.array([2, 4])
    c.support_bias = False
    calls = []

    class Servo:
        def target(self, model, data, nominal, *, active, dt):
            calls.append((nominal.copy(), active, dt))
            nominal[[1, 3]] += 0.01
            return nominal

    c.servo = Servo()
    data = SimpleNamespace(
        qpos=np.array([7.0, 8.0, 9.0, 10.0, 11.0]),
        ctrl=np.array([5.0, 6.0, 7.0, 8.0, 9.0]),
    )
    before = data.qpos.copy()
    c.command(None, data, 2.0, 0.001)
    np.testing.assert_array_equal(calls[0][0][[1, 3]], [0.1, 0.3])
    assert calls[0][1] is True
    np.testing.assert_array_equal(data.qpos, before)
    np.testing.assert_allclose(data.ctrl, [5, 6, 0.31, 8, 0.11])


def test_mounted_arm_binding_preserves_arm_object_and_native_limits(
    tmp_path, monkeypatch
):
    mj = pytest.importorskip("mujoco")
    import trimesh
    from real2sim2real.controller_actor_training import object_grasp_pilot
    from real2sim2real.controller_actor_training.contact_force_control import (
        validate_settings,
    )

    xml = ['<mujoco><compiler autolimits="true" angle="radian"/><worldbody>']
    act = []
    hands = []
    groups = ["thumb", "index", "middle", "ring", "pinky"]
    for role, side in [("right", 1), ("left", -1)]:
        xml.append(
            f'<body name="{role}_arm" pos="{side * 0.1} 0 0"><joint name="{role}_arm_joint" range="-2 2"/><geom size=".001" mass=".1" contype="0" conaffinity="0"/><body name="{role}_hand">'
        )
        act.append(
            f'<position name="{role}_arm_motor" joint="{role}_arm_joint" kp="100" forcerange="-20 20"/>'
        )
        frames = []
        targets = []
        names = []
        for i, group in enumerate(groups):
            prefix = f"{role}_{group}"
            y = (i - 2) * 0.025
            xml.append(
                f'<body name="{prefix}" pos="0 {y} 0"><joint name="{prefix}_joint" axis="0 1 0" range="-.5 .5"/><geom name="{prefix}_pad" size=".003" pos="0 0 .02"/></body>'
            )
            act.append(
                f'<position name="{prefix}_motor" joint="{prefix}_joint" kp="100" ctrlrange="-.5 .5" forcerange="-2 2"/>'
            )
            names.append(prefix + "_joint")
            targets.append([side * 0.1, y, 0.02])
            frames.append(
                dict(geom=prefix + "_pad", body=prefix, point_body_m=[0, 0, 0.02])
            )
        xml.append("</body></body>")
        hands.append(
            dict(
                accepted=True,
                root_body=role + "_hand",
                joint_names=names,
                targets_m=targets,
                pad_plan=dict(frames=frames),
                q_pregrasp=[0.1] * 5,
                q_open=[0] * 5,
                q_closed=[0.1] * 5,
            )
        )
    xml.append(
        '<body name="task_object"><freejoint name="task_object_free"/><geom type="box" size=".1 .1 .1" mass=".5"/></body></worldbody><actuator>'
        + "".join(act)
        + "</actuator></mujoco>"
    )
    model = mj.MjModel.from_xml_string("".join(xml))
    data = mj.MjData(model)
    mj.mj_forward(model, data)
    candidate = dict(hands[0], support=hands[1])
    out = tmp_path / "prepared"
    (out / "planning").mkdir(parents=True)
    (out / "planning/plan.json").write_text("{}")
    (out / "settling.json").write_text("{}")
    control = dict(
        target_force_n=dict(thumb=3, index=1.5, middle=1.5, ring=1.5, pinky=1.5),
        gain_m_per_n_s=0.001,
        max_normal_speed_m_s=0.002,
        max_joint_speed_rad_s=0.2,
        max_joint_offset_rad=0.2,
        damping_m2=1e-6,
        maximum_normal_gap_m=0.008,
        maximum_tangent_gap_m=0.015,
        close_activation_fraction=0.75,
    )
    validate_settings(control)
    parent = dict(
        candidate=candidate,
        configuration=dict(
            output=str(out),
            contact_force_feedback=control,
            support_controller=dict(
                mode="simultaneous_grasp", actuator_bias_compensation=True
            ),
            rollout=dict(approach_s=1.0, close_s=0.8, close_offset_rad=0),
        ),
    )
    monkeypatch.setattr(
        object_grasp_pilot,
        "load_plan",
        lambda c: dict(
            candidates=[candidate],
            settling=dict(settled_world_from_object=np.eye(4).tolist()),
        ),
    )
    fingers = [a for a in range(model.nu) if "arm_motor" not in model.actuator(a).name]
    before = [
        data.qpos.copy(),
        data.qvel.copy(),
        model.jnt_range.copy(),
        model.actuator_forcerange.copy(),
    ]
    c = ArmFingerFeedback(
        model,
        data,
        parent,
        dict(right=model.body("right_hand").id, left=model.body("left_hand").id),
        trimesh.creation.box(extents=[0.2, 0.2, 0.2]),
        fingers,
    )
    data.ctrl[:] = 0.123
    c.command(model, data, 0.0, 0.001)
    assert all(data.ctrl[a] == 0.123 for a in range(model.nu) if a not in fingers)
    for got, want in zip(
        (data.qpos, data.qvel, model.jnt_range, model.actuator_forcerange), before
    ):
        np.testing.assert_array_equal(got, want)

    # Real mounted-hand pressure injection retains sole finger ownership.
    from real2sim2real.controller_actor_training.residual_pressure import CHANNELS

    c.command(model, data, 2.0, 0.001, pressure_offsets=dict.fromkeys(CHANNELS, -0.5))
    assert all(data.ctrl[a] == 0.123 for a in range(model.nu) if a not in fingers)
    for pad in c.servo.pads:
        assert pad["target_n"] == control["target_force_n"][pad["group"]] - 0.5
    for got, want in zip(
        (data.qpos, data.qvel, model.jnt_range, model.actuator_forcerange), before
    ):
        np.testing.assert_array_equal(got, want)

    # Reading policy observations must not perturb live solver state or commands.
    from real2sim2real.controller_actor_training.arm_residual_observation import (
        ArmResidualObservation,
    )
    from real2sim2real.controller_actor_training.residual_pressure import (
        PressureResidual,
    )

    parent["records"] = [
        dict(
            phase="hold",
            reference_position_m=[0, 0, 0],
            reference_rotation=np.eye(3).tolist(),
        )
    ] * 2
    observer = ArmResidualObservation(
        model,
        data,
        c,
        dict(right=model.body("right_hand").id, left=model.body("left_hand").id),
        parent,
        np.array([0, 0.02]),
    )
    controller = PressureResidual(
        dict(
            policy_period_s=0.02,
            max_offset_n=1.0,
            max_rate_n_s=5.0,
            active_phases=["hold"],
        )
    )
    fields = (
        "qpos",
        "qvel",
        "ctrl",
        "qacc",
        "qacc_warmstart",
        "qfrc_bias",
        "actuator_force",
    )
    original = {name: getattr(data, name).copy() for name in fields}
    packet = observer.capture(0.0, 0, 0.0, controller)
    assert np.isfinite(packet["observation"]).all()
    for name in fields:
        np.testing.assert_array_equal(getattr(data, name), original[name])
