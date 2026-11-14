from copy import deepcopy

import numpy as np
import pytest
from real2sim2real.agentic_policy.manipulation import (
    ManipulationSkills,
    pose,
    relative_pose,
)

mujoco = pytest.importorskip("mujoco")
from real2sim2real.controller_actor_training.continuous_policy import ContinuousPolicy


def session(tmp_path, name, obstacle=False):
    root = tmp_path / name
    root.mkdir()
    # Scalar arm/finger fixture, not evidence of a UR/Sharpa grasp success.
    obstacle_xml = (
        '<body pos=".7 .075 .05"><geom name="obstacle" type="sphere" size=".02"/></body>'
        if obstacle
        else ""
    )
    xml = f"""<mujoco><option timestep="0.001" integrator="implicitfast"/><worldbody>
    <geom name="floor" type="plane" size="2 2 .1"/>{obstacle_xml}
    <body name="base" pos="0 0 1"><body name="arm"><joint name="arm_joint" axis="0 0 1" range="-90 90"/>
    <geom type="sphere" size=".03" mass="1" contype="0" conaffinity="0"/>
    <body name="hand" pos=".2 0 0"><joint name="finger_joint" range="-90 90"/>
    <geom name="pad" type="sphere" size=".02" mass=".1" contype="0" conaffinity="0"/></body></body></body>
    <body name="object" pos=".7 0 .05" quat="0 1 0 0"><freejoint/><geom type="sphere" size=".05" mass=".2"/></body>
    </worldbody><actuator><position joint="arm_joint" kp="100" kv="10" ctrlrange="-1.57 1.57"/>
    <position joint="finger_joint" kp="100" kv="10" ctrlrange="-1.57 1.57"/></actuator></mujoco>"""
    (root / "scene.xml").write_text(xml)
    model = mujoco.MjModel.from_xml_path(str(root / "scene.xml"))
    np.savez(root / "initial.npz", qpos=model.qpos0, qvel=np.zeros(model.nv))
    return ContinuousPolicy(
        {
            "scene": str(root / "scene.xml"),
            "initial": str(root / "initial.npz"),
            "base_body": "base",
            "base_placement": {"position_m": [0, 0, 1], "yaw_rad": 0},
            "arm_joints": ["arm_joint"],
            "hand_root": "hand",
            "object_body": "object",
            "pad_frames": [
                {
                    "body": "hand",
                    "geom": "pad",
                    "point_body_m": [0, 0, 0],
                    "normal_body": [0, 0, 1],
                }
            ],
            "support_geoms": ["floor"],
            "feedback_s": 0.005,
            "sample_s": 0.02,
            "horizon_s": 2.0,
            "penetration_limit_m": 0.003,
            "wall_deadline_s": 30,
            "velocity_limits_rad_s": {"arm_joint": 3.0, "finger_joint": 1.0},
            "task": {
                "support_dwell_s": 0.15,
                "bay_xy_m": [0.7, 0],
                "bay_radius_m": 0.035,
                "inverted_angle_deg": 30,
                "minimum_support_fraction": 0.5,
                "maximum_hand_force_n": 0.1,
                "maximum_object_speed_m_s": 0.02,
                "stable_window_s": 0.1,
                "minimum_window_samples": 5,
            },
        }
    )


def test_acquisition_is_exact_existing_motion_without_pause_or_false_hold_claim(
    tmp_path,
):
    a, b = session(tmp_path, "adapted"), session(tmp_path, "direct")
    q = {"arm_joint": 0.1, "finger_joint": 0.05}
    skills = ManipulationSkills(a)
    result = skills.acquire_grasp([q], [0.5], [0.0])
    b.execute_joint_path([q], [0.5])
    b.set_pressure([0.0])
    assert result["executed"] and not result["loaded_hold_verified"]
    np.testing.assert_array_equal(a.data.qpos, b.data.qpos)
    np.testing.assert_array_equal(a.data.qvel, b.data.qvel)
    np.testing.assert_array_equal(a.commands, b.commands)
    assert a.tick == b.tick == 500
    check = skills.verify_grasp(0.1)
    assert not check["held"]
    assert not check["checks"]["contact_count"]
    assert not check["checks"]["off_support"]
    assert a.tick == 600


def test_bad_pressure_and_paths_fail_before_any_motion(tmp_path):
    s = session(tmp_path, "invalid")
    skills = ManipulationSkills(s)
    initial = s.data.qpos.copy()
    for targets in [[float("nan")], [6.0], [], [-0.1]]:
        with pytest.raises(ValueError):
            skills.acquire_grasp([dict(zip(s.names, s.home))], [0.1], targets)
    with pytest.raises(ValueError):
        skills.acquire_grasp([{"arm_joint": 2, "finger_joint": 0}], [0.1], [0.0])
    assert s.tick == 0
    np.testing.assert_array_equal(initial, s.data.qpos)


def test_transport_planning_is_read_only_and_stale_plans_cannot_execute(tmp_path):
    s = session(tmp_path, "plan")
    skills = ManipulationSkills(s)
    before = s.data.qpos.copy(), s.data.qvel.copy()
    target = pose(s.observe(), "object").tolist()
    planned = skills.plan_transport([target], [0.2])
    assert planned["admitted"]
    np.testing.assert_array_equal(before[0], s.data.qpos)
    np.testing.assert_array_equal(before[1], s.data.qvel)
    assert s.tick == 0
    result = skills.execute_transport(planned["plan_id"])
    assert result["executed"] and s.tick == 200
    assert not skills.execute_transport(planned["plan_id"])["executed"]
    assert s.tick == 200


def test_contact_aware_transport_requires_evidence_and_guard_does_not_step_on_failure(
    tmp_path,
):
    s = session(tmp_path, "guard")
    s.config["transport_screen_mode"] = "contact_aware"
    skills = ManipulationSkills(s)
    planned = skills.plan_transport([pose(s.observe(), "object").tolist()], [0.2])
    assert not planned["admitted"]
    assert "evidence" in planned["contact_aware_screen"]["reason"]
    assert s.tick == 0
    with pytest.raises(RuntimeError, match="diagnostic_guard"):
        s._execute_screened_path(
            [dict(zip(s.names, s.nominal))],
            [0.1],
            feedback_guard=lambda obs: "diagnostic_guard",
        )
    assert s.tick == 0 and s.fault == "diagnostic_guard"


def test_loaded_preview_fork_preserves_live_state_and_matches_real_actuator_loop(
    tmp_path,
):
    s = session(tmp_path, "fork")
    from real2sim2real.controller_actor_training.pad_pressure_correction import (
        PadPressureCorrection,
    )

    s.pressure = PadPressureCorrection(
        [[0]],
        s.model.jnt_range[s.hand["joints"]],
        {
            "target_force_n": 1.0,
            "gain_m_per_n_s": 0.002,
            "max_normal_speed_m_s": 0.003,
            "max_joint_speed_rad_s": 0.3,
            "max_joint_offset_rad": 0.06,
            "damping_m2": 1e-6,
        },
    )
    s.pressure_targets = np.array([1.0])
    s.pressure.offset[:] = 0.01
    signature = mujoco.mjtState.mjSTATE_INTEGRATION
    before = np.empty(mujoco.mj_stateSize(s.model, signature))
    mujoco.mj_getState(s.model, s.data, before, signature)
    fork = s._fork_for_preview()
    assert fork.pressure is not s.pressure
    assert not np.shares_memory(fork.pressure.offset, s.pressure.offset)
    target = {"arm_joint": 0.05, "finger_joint": 0.02}
    fork._execute_screened_path([target], [0.2])
    current = before.copy()
    mujoco.mj_getState(s.model, s.data, current, signature)
    np.testing.assert_array_equal(current, before)
    assert s.tick == 0 and len(s.history) == 1
    s._execute_screened_path([target], [0.2])
    np.testing.assert_array_equal(fork.data.qpos, s.data.qpos)
    np.testing.assert_array_equal(fork.data.qvel, s.data.qvel)
    np.testing.assert_array_equal(fork.data.ctrl, s.data.ctrl)
    np.testing.assert_array_equal(fork.pressure.offset, s.pressure.offset)
    assert fork.tick == s.tick == 200


def test_contact_aware_hold_rejects_missing_loaded_evidence_without_stepping(tmp_path):
    s = session(tmp_path, "loaded_hold")
    s.config["transport_screen_mode"] = "contact_aware_preview"
    skills = ManipulationSkills(s)
    skills.anchor = relative_pose(s.observe())
    before = s.data.qpos.copy()
    result = skills.verify_grasp(0.2)
    assert not result["held"] and not result["hold"]["executed"]
    assert s.tick == 0
    np.testing.assert_array_equal(s.data.qpos, before)


def test_carried_screen_detects_object_obstacle_collision_omitted_by_fixed_object_screen(
    tmp_path,
):
    s = session(tmp_path, "carried", obstacle=True)
    skills = ManipulationSkills(s)
    q = {"arm_joint": 0.1, "finger_joint": 0.0}
    before = s.data.qpos.copy()
    assert s.screen_path([q], [0.5])["admitted"]
    result = skills._carried_screen([q], [0.5], relative_pose(s.observe()))
    assert not result["admitted"]
    assert "obstacle" in result["witness"]
    np.testing.assert_array_equal(before, s.data.qpos)
    assert s.tick == 0


def test_pad_fitting_is_read_only_and_returns_no_acquisition_claim(tmp_path):
    s = session(tmp_path, "fit")
    skills = ManipulationSkills(s)
    obs = s.observe()
    q = s.data.qpos.copy()
    result = skills.fit_grasp(
        pose(obs, "wrist").tolist(),
        obs["joints_rad"],
        [0],
        obs["pad_points_world_m"],
        obs["pad_normals_world"],
    )
    assert result["fit_pass"] and result["path_admission_required"]
    assert not result["loaded_hold_verified"]
    np.testing.assert_array_equal(q, s.data.qpos)
    assert s.tick == 0


def test_release_requires_correct_bay_and_dwell_and_keeps_the_clock(tmp_path):
    s = session(tmp_path, "place")
    skills = ManipulationSkills(s)
    home = dict(zip(s.names, s.home))
    open_q = home | {"finger_joint": 0.05}
    with pytest.raises(ValueError, match="tick-aligned"):
        skills.place_and_release([home], [0.05], open_q, 0.1000001)
    assert s.tick == 0
    # Not enough support dwell; no hidden wait is inserted.
    result = skills.place_and_release([home], [0.05], open_q, 0.1)
    assert not result["released"] and s.tick == 50
    result = skills.place_and_release([home], [0.2], open_q, 0.1)
    assert result["released"] and s.tick == 350
    assert skills.anchor is None
    assert skills.verify_placement()["placed"]
    assert s.tick == 350
    s.config["task"]["bay_xy_m"] = [3, 3]  # fixture tests wrong-bay rejection
    result = skills.place_and_release([open_q], [0.05], home, 0.1)
    assert not result["released"] and s.tick == 400
    assert not skills.verify_placement()["placed"]


def test_hold_detects_rotation_and_intermittent_contact_loss(tmp_path, monkeypatch):
    s = session(tmp_path, "hold")
    skills = ManipulationSkills(s)
    start = s.observe()
    start.update(pad_forces_n=[1.0, 1.0], support_fraction=0.0)
    end = deepcopy(start)
    end["time_s"] = 0.1
    middle = deepcopy(start)
    middle.update(time_s=0.05, pad_forces_n=[0.0, 0.0])
    middle["object_rotation"] = (
        np.array([[0.0, -1, 0], [1, 0, 0], [0, 0, 1]])
        @ np.array(start["object_rotation"])
    ).tolist()
    s.history = [middle]
    skills.anchor = relative_pose(start)
    observed = iter([start, end])
    monkeypatch.setattr(s, "observe", lambda: next(observed))
    monkeypatch.setattr(s, "hold", lambda duration: {"executed": True})
    result = skills.verify_grasp(0.1)
    assert not result["held"]
    assert not result["checks"]["rotation"] and not result["checks"]["contact_count"]
    assert result["metrics"]["rotation_drift_deg"] == pytest.approx(90)


@pytest.mark.parametrize("fault", [None, "contact_loss", "support", "translation"])
def test_hold_gate_accepts_complete_evidence_and_rejects_midwindow_failures(
    tmp_path, monkeypatch, fault
):
    s = session(tmp_path, "evidence")
    skills = ManipulationSkills(s)
    start = s.observe()
    start.update(pad_forces_n=[1.0, 1.0], support_fraction=0.0)
    rows = [deepcopy(start) | {"time_s": float(t)} for t in np.linspace(0, 0.1, 6)]
    if fault == "contact_loss":
        rows[2]["pad_forces_n"] = [0.0, 0.0]
    elif fault == "support":
        rows[2]["support_fraction"] = 1.0
    elif fault == "translation":
        rows[2]["object_position_m"][0] += 0.02
    s.history = rows[1:-1]
    skills.anchor = relative_pose(start)
    observed = iter([rows[0], rows[-1]])
    monkeypatch.setattr(s, "observe", lambda: next(observed))
    monkeypatch.setattr(s, "hold", lambda duration: {"executed": True})
    assert skills.verify_grasp(0.1)["held"] is (fault is None)
