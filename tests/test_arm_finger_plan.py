from copy import deepcopy
import json

import numpy as np
import pytest

from real2sim2real.artifacts import digest
from real2sim2real.controller_actor_training.arm_finger_plan import (
    validate_hand_change,
    validate_stationary,
)


def hand():
    return dict(
        joint_names=["fixture_x", "index_PIP"],
        root_body="hand",
        fixture_prefix="fixture_",
        **{k: [0.0, 0.2] for k in ("q_pregrasp", "q_open", "q_grasp", "q_closed")},
    )


def test_finger_plan_preserves_wrist_identity_acquisition_and_bounded_fingers():
    a = hand()
    b = deepcopy(a)
    b["q_closed"][1] += 0.2
    poses = {
        k: (np.zeros(3), np.eye(3))
        for k in ("q_pregrasp", "q_open", "q_grasp", "q_closed")
    }
    validate_hand_change(a, b, poses, poses)
    b["q_open"][1] += 0.01
    with pytest.raises(ValueError, match="acquisition"):
        validate_hand_change(a, b, poses, poses)
    b = hand()
    b["q_closed"][1] += 0.4
    with pytest.raises(ValueError, match="bounded"):
        validate_hand_change(a, b, poses, poses)
    b = hand()
    shifted = deepcopy(poses)
    shifted["q_closed"][0][0] = 0.002
    with pytest.raises(ValueError, match="wrist"):
        validate_hand_change(a, b, poses, shifted)


def test_stationary_admission_pins_clock_posture_physics_and_complete_hold(tmp_path):
    source = tmp_path / "source.json"
    source.write_text(json.dumps(dict(records=[dict(time_s=0), dict(time_s=5)])))
    audit = tmp_path / "audit.json"
    audit.write_text(
        json.dumps(
            dict(config=dict(trial=str(source)), sources={str(source): digest(source)})
        )
    )
    path = tmp_path / "result.json"
    states = tmp_path / "states.npz"
    states.write_bytes(b"source state archive")
    scene = tmp_path / "scene.xml"
    scene.write_text("<mujoco/>")
    settings = dict(
        audit=str(audit),
        audit_sha256=digest(audit),
        control_period_s=0.001,
        timestep_s=0.000125,
        finger_plan=dict(sha256="unchanged", candidate_id=9000),
        motion_condition="stationary_hold",
        output="stationary",
    )
    report = dict(
        schema="real2sim2real.arm-replay-pilot/1",
        success=True,
        motion_condition="stationary_hold",
        states_sha256=digest(states),
        scene_sha256=digest(scene),
        config=settings,
        records=[dict(time_s=0), dict(time_s=5)],
    )

    def proposed():
        path.write_text(json.dumps(report))
        return dict(
            settings,
            motion_condition="demonstrated",
            output="moving",
            validated_stationary=dict(result=str(path), sha256=digest(path)),
        )

    assert validate_stationary(proposed())["sha256"] == digest(path)
    report["residual_adapter"] = {"all_actions_zero": False}
    with pytest.raises(ValueError, match="policy-aware"):
        validate_stationary(proposed())
    del report["residual_adapter"]
    changed = proposed()
    changed["control_period_s"] = 0.002
    with pytest.raises(ValueError, match="retune"):
        validate_stationary(changed)
    report["records"][-1]["time_s"] = 4.0
    with pytest.raises(ValueError, match="complete"):
        validate_stationary(proposed())
    report["records"][-1]["time_s"] = 5.0
    report["success"] = False
    with pytest.raises(ValueError, match="Passing"):
        validate_stationary(proposed())


def test_finger_path_screen_leaves_physical_state_and_native_limits_unchanged():
    mj = pytest.importorskip("mujoco")
    from types import SimpleNamespace
    from real2sim2real.controller_actor_training.arm_finger_plan import path_screen

    model = mj.MjModel.from_xml_string("""<mujoco><compiler autolimits="true"/><worldbody>
    <geom name="wall" type="box" size=".01 .01 .01" pos="1 1 1"/>
    <body name="arm" pos="0 0 .2"><joint name="arm_joint" axis="0 0 1" range="-1 1"/><geom name="arm_geom" size=".01"/>
    <body name="finger" pos=".05 0 0"><joint name="finger_joint" axis="0 1 0" range="-1 1"/><geom name="pad" size=".01"/></body></body>
    <body name="object" pos="2 0 .2"><freejoint/><geom name="object_geom" size=".02"/></body>
    </worldbody></mujoco>""")
    data = mj.MjData(model)
    mj.mj_forward(model, data)
    before = (data.qpos.copy(), data.qvel.copy(), model.jnt_range.copy())
    live = SimpleNamespace(
        hands=[
            dict(
                qids=np.array([1]),
                waypoints=[np.zeros(1), np.zeros(1), np.array([0.1])],
            )
        ],
        rollout=dict(approach_s=1.0, close_s=0.8),
    )
    args = (
        model,
        data,
        np.array([[0.0], [0.1], [0.2]]),
        [0.0, 1.0, 2.0],
        np.array([0]),
        live,
        {model.geom("arm_geom").id},
        {model.geom("arm_geom").id, model.geom("pad").id},
        {model.geom("object_geom").id},
        dict(arm_penetration_m=0.001, robot_penetration_m=0.003),
    )
    result = path_screen(*args)
    assert result["accepted"] and result["samples"] == 3
    for actual, original in zip((data.qpos, data.qvel, model.jnt_range), before):
        np.testing.assert_array_equal(actual, original)
    model.geom_pos[model.geom("wall").id] = [0.05, 0, 0.2]
    assert not path_screen(*args)["accepted"]
