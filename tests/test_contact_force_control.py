from copy import deepcopy
import json

import numpy as np
import pytest
import trimesh

from real2sim2real.controller_actor_training.contact_force_control import (
    normal_force_step,
    validate_settings,
)


def settings():
    return dict(
        target_force_n=dict(thumb=3.0, index=1.5, middle=1.5, ring=1.5, pinky=1.5),
        gain_m_per_n_s=0.001,
        max_normal_speed_m_s=0.002,
        max_joint_speed_rad_s=0.2,
        max_joint_offset_rad=0.2,
        damping_m2=1e-6,
        maximum_normal_gap_m=0.008,
        maximum_tangent_gap_m=0.015,
        close_activation_fraction=0.75,
    )


def test_force_servo_presses_when_low_and_releases_when_high():
    s = settings()
    limits = np.array([[-0.5, 0.5]])
    q, offset = normal_force_step([0], [0], [0.01], 0, 2, 0.01, limits, s)
    assert q[0] < 0 and 0.01 * q[0] < 0
    assert abs(q[0]) <= 0.2 * 0.01
    release, _ = normal_force_step([0], [0], [0.01], 4, 2, 0.01, limits, s)
    assert release[0] > 0
    np.testing.assert_array_equal(offset, q)


def test_joint_stop_antiwindup_uses_remaining_joints_and_caps_integral():
    s = settings()
    limits = np.array([[0, 0.5], [-0.1, 0.5]])
    q, offset = normal_force_step([0, 0], [0, 0], [0.01, 0.01], 0, 3, 0.01, limits, s)
    assert q[0] == 0 and q[1] < 0
    for _ in range(1000):
        q, offset = normal_force_step(
            [0, 0], offset, [0.01, 0.01], 0, 3, 0.01, limits, s
        )
    assert q[1] == pytest.approx(-0.1) and offset[1] == pytest.approx(-0.1)
    assert np.linalg.norm(offset) <= s["max_joint_offset_rad"]
    for bad in [float("nan"), -1]:
        with pytest.raises(ValueError):
            normal_force_step([0], [0], [0.01], bad, 2, 0.01, [[-1, 1]], s)
    with pytest.raises(ValueError):
        normal_force_step([2], [0], [0.01], 1, 2, 0.01, [[-1, 1]], s)


def test_tangent_step_corrects_sliding_without_reversing_pressure():
    s = settings()
    normal = np.array([0.01, 0, 0])
    tangent = np.diag([0, 0.01, 0.01])
    original = normal.copy()
    q, offset = normal_force_step(
        np.zeros(3),
        np.zeros(3),
        normal,
        1,
        2,
        0.01,
        np.array([[-1, 1]] * 3),
        s,
        tangent_jacobian=tangent,
        tangent_velocity=[0, 0.001, -0.001],
    )
    assert q[0] < 0 and q[1] > 0 and q[2] < 0
    assert np.linalg.norm(offset) <= 0.01 * s["max_joint_speed_rad_s"] + 1e-12
    np.testing.assert_array_equal(normal, original)
    blocked, _ = normal_force_step(
        np.zeros(3),
        np.zeros(3),
        normal,
        1,
        2,
        0.01,
        np.array([[0, 1], [-1, 1], [-1, 1]]),
        s,
        tangent_jacobian=tangent,
        tangent_velocity=[0, 0.001, -0.001],
    )
    assert blocked[0] == 0 and blocked[1] > 0 and blocked[2] < 0
    s.update(tangent_gain_s_inv=1, max_tangent_speed_m_s=0.002)
    validate_settings(s)
    s["normal_observation"] = "solved_contact"
    with pytest.raises(ValueError, match="tangent"):
        validate_settings(s)


def test_controller_settings_reject_unbounded_or_incomplete_force_targets():
    s = settings()
    validate_settings(s)
    for field, value in [
        ("max_joint_offset_rad", 0.6),
        ("gain_m_per_n_s", float("nan")),
        ("close_activation_fraction", 0.1),
    ]:
        bad = deepcopy(s)
        bad[field] = value
        with pytest.raises(ValueError):
            validate_settings(bad)
    s["target_force_n"]["thumb"] = 11
    with pytest.raises(ValueError):
        validate_settings(s)


def test_pad_force_measurement_excludes_unrelated_object_contact():
    mj = pytest.importorskip("mujoco")
    from real2sim2real.controller_actor_training.contact_force_control import (
        object_pad_forces,
        object_pad_observations,
    )

    model = mj.MjModel.from_xml_string("""<mujoco><option timestep=".002"/><worldbody>
    <body name="hand"><geom name="pad" type="box" size=".4 .4 .01"/></body>
    <body name="task_object" pos="-.1 0 .04"><freejoint/><geom size=".02" mass=".2"/></body>
    <body name="clutter" pos=".1 0 .04"><freejoint/><geom size=".02" mass="1"/></body>
    </worldbody></mujoco>""")
    data = mj.MjData(model)
    for _ in range(500):
        mj.mj_step(model, data)
    values = object_pad_forces(
        model, data, model.body("task_object").id, [model.geom("pad").id]
    )
    assert values[model.geom("pad").id] == pytest.approx(0.2 * 9.81, abs=0.02)
    row = object_pad_observations(
        model, data, model.body("task_object").id, [model.geom("pad").id]
    )[model.geom("pad").id]
    np.testing.assert_allclose(row["outward_normal_world"], [0, 0, -1], atol=1e-5)
    assert row["point_world_m"][2] == pytest.approx(0.01, abs=0.001)


def test_stationary_promotion_pins_controller_candidate_completion_and_states(tmp_path):
    from real2sim2real.artifacts import digest
    from real2sim2real.controller_actor_training.bilateral_evaluate import (
        validate_stationary_promotion,
    )

    directory = tmp_path / "candidate-0"
    directory.mkdir()
    path = directory / "result.json"
    states = directory / "states.npz"
    states.write_bytes(b"pinned test archive")
    base = dict(
        contact_force_feedback=settings(),
        robot={"limit": 2},
        planning_directory="new",
        reference_guidance={"path": "ref"},
    )
    before = deepcopy(base)
    before.pop("reference_guidance")
    before["planning_directory"] = "old"
    candidate = {"id": 0}
    r = dict(
        success=True,
        simulation_final_time_s=10.0,
        simulation_requested_duration_s=10.0,
        states_sha256=digest(states),
        candidate=candidate,
        configuration=before,
    )

    limits = dict(position_m=0.02, rotation_deg=10)
    (tmp_path / "protocol.json").write_text(
        json.dumps(
            dict(motion_condition="stationary_diagnostic", fidelity_limits=limits)
        )
    )

    def config():
        path.write_text(json.dumps(r))
        return dict(
            fidelity_limits=limits,
            validated_stationary=dict(result=str(path), sha256=digest(path)),
        )

    validate_stationary_promotion(config(), base, candidate)
    altered = config()
    altered["fidelity_limits"] = dict(position_m=0.03, rotation_deg=10)
    with pytest.raises(ValueError, match="acceptance"):
        validate_stationary_promotion(altered, base, candidate)
    bad = deepcopy(base)
    bad["contact_force_feedback"]["max_joint_offset_rad"] = 0.3
    with pytest.raises(ValueError, match="retune"):
        validate_stationary_promotion(config(), bad, candidate)
    bad = deepcopy(base)
    bad["object_pose_feedback"] = {"rotation_gain_s_inv": 1}
    with pytest.raises(ValueError, match="retune"):
        validate_stationary_promotion(config(), bad, candidate)
    r["simulation_final_time_s"] = 5
    with pytest.raises(ValueError, match="complete"):
        validate_stationary_promotion(config(), base, candidate)
    r["simulation_final_time_s"] = 10
    r["success"] = False
    with pytest.raises(ValueError):
        validate_stationary_promotion(config(), base, candidate)


@pytest.mark.parametrize("payload_adaptation", [False, True])
def test_force_feedback_owns_fingers_and_preserves_physical_state_and_limits(
    payload_adaptation,
):
    mj = pytest.importorskip("mujoco")
    from real2sim2real.controller_actor_training.contact_force_control import (
        BilateralContactForce,
    )

    fingers = ["thumb", "index", "middle", "ring", "pinky"]
    xml = ["<mujoco><worldbody>"]
    actuators = []
    hands = []
    for role, side in [("primary", 1), ("secondary", -1)]:
        root = role + "_root"
        prefix = role + "_fixture_"
        names = [prefix + a for a in "xyz"]
        frames = []
        targets = []
        xml.append(f'<body name="{root}" pos="{side * 0.1} 0 0">')
        for axis, vector in zip("xyz", ["1 0 0", "0 1 0", "0 0 1"]):
            xml.append(
                f'<joint name="{prefix + axis}" type="slide" axis="{vector}" range="-.3 .3"/>'
            )
        xml.append('<geom size=".001" mass=".01" contype="0" conaffinity="0"/>')
        for i, finger in enumerate(fingers):
            name = role + "_" + finger
            names.append(name + "_hinge")
            point = [0, 0, 0.02]
            y = (i - 2) * 0.025
            xml.append(
                f'<body name="{name}" pos="0 {y} 0"><joint name="{name}_hinge" axis="0 1 0" range="-.5 .5"/><geom name="{name}_pad" size=".003" pos="0 0 .02"/></body>'
            )
            actuators.append(
                f'<position joint="{name}_hinge" kp="100" ctrlrange="-.5 .5" forcerange="-2 2"/>'
            )
            frames.append(dict(geom=name + "_pad", body=name, point_body_m=point))
            targets.append([side * 0.1, y, 0.02])
        xml.append("</body>")
        hands.append(
            dict(
                root_body=root,
                joint_names=names,
                pad_plan=dict(frames=frames),
                targets_m=targets,
            )
        )
    xml.append(
        '<body name="task_object"><freejoint name="task_object_free"/><geom type="box" size=".1 .1 .1" mass=".5"/></body></worldbody><actuator>'
        + "".join(actuators)
        + "</actuator></mujoco>"
    )
    model = mj.MjModel.from_xml_string("".join(xml))
    data = mj.MjData(model)
    mj.mj_forward(model, data)
    candidate = dict(hands[0], support=hands[1])
    mesh = trimesh.creation.box(extents=[0.2, 0.2, 0.2])
    control = settings()
    if payload_adaptation:
        control["payload_adaptation"] = dict(
            reference_weight_n=4.905,
            weight_filter_s=0.15,
            share_filter_s=0.15,
            minimum_weight_ratio=0.5,
            maximum_weight_ratio=1.25,
            target_secondary_fraction=0.5,
            share_gain=0.5,
            maximum_share_correction=0.15,
            maximum_scale_rate_s_inv=1.0,
            maximum_acceleration_m_s2=3.0,
        )
    controller = BilateralContactForce(model, data, control, candidate, np.eye(4), mesh)
    q = data.qpos.copy()
    ranges = model.jnt_range.copy()
    forces = model.actuator_forcerange.copy()
    np.testing.assert_array_equal(
        controller.target(model, data, q, active=False, dt=0.002), q
    )
    command = controller.target(model, data, q, active=True, dt=0.002)
    owned = np.concatenate([p["qids"] for p in controller.pads])
    unowned = np.setdiff1d(np.arange(model.nq), owned)
    np.testing.assert_array_equal(command[unowned], q[unowned])
    np.testing.assert_array_equal(data.qpos, q)
    np.testing.assert_array_equal(model.jnt_range, ranges)
    np.testing.assert_array_equal(model.actuator_forcerange, forces)
    assert np.linalg.norm(command[owned] - q[owned]) > 0


def test_cross_study_review_names_cannot_overwrite_same_candidate_media(tmp_path):
    from real2sim2real.controller_actor_training.object_grasp_review import (
        unique_trial_media_names,
    )

    paths = [tmp_path / "baseline" / "candidate-0", tmp_path / "force" / "candidate-0"]
    names = unique_trial_media_names(paths)
    assert names == ["baseline--candidate-0", "force--candidate-0"]
    assert unique_trial_media_names([tmp_path / "one", tmp_path / "two"]) == [
        "one",
        "two",
    ]
    with pytest.raises(ValueError, match="overwrite"):
        unique_trial_media_names([paths[0], paths[0]])


def test_collision_witness_preserves_pair_and_excludes_nonrobot_contacts():
    from types import SimpleNamespace
    from real2sim2real.controller_actor_training.object_grasp_audit import (
        penetration_witnesses,
    )

    model = SimpleNamespace(
        geom=lambda i: SimpleNamespace(name=["pad", "object", "table"][i]),
        body=lambda i: SimpleNamespace(name=["hand", "object", "world"][i]),
        geom_bodyid=[0, 1, 2],
    )

    def contact(a, b, distance):
        return SimpleNamespace(geom1=a, geom2=b, dist=distance, pos=[1, 2, 3])

    data = SimpleNamespace(
        contact=[contact(1, 0, -0.004), contact(1, 2, -0.008), contact(0, 2, 0.001)]
    )
    rows = penetration_witnesses(model, data, {0})
    assert len(rows) == 1
    assert rows[0]["geoms"] == ["pad", "object"]
    assert rows[0]["penetration_m"] == 0.004
    assert rows[0]["position_world_m"] == [1, 2, 3]
