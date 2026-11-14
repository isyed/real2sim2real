import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
from real2sim2real.controller_actor_training.continuous_policy import ContinuousPolicy


def make_session(tmp_path, name):
    p = tmp_path / name
    p.mkdir()
    xml = """<mujoco><option timestep="0.001" integrator="implicitfast"/><worldbody>
    <geom name="floor" type="plane" size="2 2 .1"/>
    <body name="base" pos="0 0 1"><body name="arm"><joint name="arm_joint" range="-90 90"/><geom type="sphere" size=".03" mass="1" contype="0" conaffinity="0"/>
    <body name="hand" pos=".2 0 0"><joint name="finger_joint" range="-90 90"/><geom name="pad" type="sphere" size=".02" mass=".1" contype="0" conaffinity="0"/></body></body></body>
    <body name="object" pos=".7 0 .05"><freejoint/><geom type="sphere" size=".05" mass=".2"/></body>
    </worldbody><actuator><position joint="arm_joint" kp="100" kv="10" ctrlrange="-1.57 1.57"/><position joint="finger_joint" kp="100" kv="10" ctrlrange="-1.57 1.57"/></actuator></mujoco>"""
    (p / "scene.xml").write_text(xml)
    m = mujoco.MjModel.from_xml_path(str(p / "scene.xml"))
    np.savez(p / "initial.npz", qpos=m.qpos0, qvel=np.zeros(m.nv))
    return ContinuousPolicy(
        {
            "scene": str(p / "scene.xml"),
            "initial": str(p / "initial.npz"),
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
            "horizon_s": 1.0,
            "penetration_limit_m": 0.003,
            "wall_deadline_s": 30,
            "velocity_limits_rad_s": {"arm_joint": 3.0, "finger_joint": 1.0},
            "task": {"support_dwell_s": 0.15},
        }
    )


def test_invalid_commands_do_not_advance_or_mutate_physics(tmp_path):
    e = make_session(tmp_path, "atomic")
    q = e.data.qpos.copy()
    t = e.data.time
    for command, duration in [
        ({"object": [0, 0, 1]}, 0.1),
        ({"arm_joint": 0, "finger_joint": 9}, 0.1),
        ({"arm_joint": 1, "finger_joint": 0}, 0.01),
    ]:
        with pytest.raises(ValueError):
            e.execute_joint_path([command], [duration])
        np.testing.assert_array_equal(q, e.data.qpos)
        assert e.data.time == t
    assert e.release(dict(zip(e.names, e.home)), 0.1)["released"] is False


def test_hold_partition_and_observation_preserve_continuous_state(tmp_path):
    a = make_session(tmp_path, "one")
    b = make_session(tmp_path, "many")
    a.hold(0.2)
    for _ in range(4):
        b.observe()
        b.hold(0.05)
    np.testing.assert_array_equal(a.data.qpos, b.data.qpos)
    np.testing.assert_array_equal(a.data.qvel, b.data.qvel)
    assert a.tick == b.tick == 200
    assert a.data.time == b.data.time
    assert a.data.qpos[-7:].tolist() == b.data.qpos[-7:].tolist()


def test_nonzero_command_uses_dynamics_and_budget(tmp_path):
    e = make_session(tmp_path, "move")
    target = {"arm_joint": 0.1, "finger_joint": 0.05}
    assert e.execute_joint_path([target], [0.5])["executed"]
    assert e.data.time == pytest.approx(0.5)
    assert e.data.qpos[e.qids[0]] > 0.05
    with pytest.raises(ValueError):
        e.hold(0.6)


def test_save_captures_final_feedback_tick_between_regular_samples(tmp_path):
    e = make_session(tmp_path, "final")
    e.hold(0.025)
    e.save(tmp_path / "saved")
    with np.load(tmp_path / "saved/states.npz") as a:
        assert a["timestamps_s"][-1] == pytest.approx(0.025)
        np.testing.assert_array_equal(a["qpos"][-1], e.data.qpos)


def test_supported_release_advances_existing_clock_without_object_reset(tmp_path):
    e = make_session(tmp_path, "release")
    e.hold(0.25)
    target = dict(zip(e.names, e.home))
    target["finger_joint"] = 0.05
    result = e.release(target, 0.25)
    assert result["released"]
    assert e.tick == 500
    assert e.data.time == pytest.approx(0.5)
    assert e.data.qpos[e.qids[1]] > 0.02
