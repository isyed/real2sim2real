import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from real2sim2real.controller_actor_training.fixture_hold import Commands, clock_counts


def test_clock_decouples_feedback_from_physics_and_covers_endpoint():
    assert clock_counts(0.000125, 0.001, 0.02, 16.744416) == (8, 160, 133956)
    assert clock_counts(0.00025, 0.001, 0.02, 16.744416) == (4, 80, 66978)
    with pytest.raises(ValueError, match="integer"):
        clock_counts(0.0003, 0.001, 0.02, 10)
    with pytest.raises(ValueError, match="finite"):
        clock_counts(np.nan, 0.001, 0.02, 10)


def fixture_model():
    mujoco = pytest.importorskip("mujoco")
    bodies = []
    for name, pos in [("right", "-.2 0 .1"), ("left", ".2 0 .1")]:
        joints = "".join(
            f'<joint name="{name}_{axis}" type="slide" axis="{vector}" range="-1 1"/>'
            for axis, vector in [("x", "1 0 0"), ("y", "0 1 0"), ("z", "0 0 1")]
        )
        joints += "".join(
            f'<joint name="{name}_r{axis}" type="hinge" axis="{vector}" range="-180 180"/>'
            for axis, vector in [("x", "1 0 0"), ("y", "0 1 0"), ("z", "0 0 1")]
        )
        bodies.append(
            f'<body name="{name}" pos="{pos}">{joints}<geom type="sphere" size=".01" mass="1"/></body>'
        )
    model = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody>"
        + "".join(bodies)
        + '<body name="task_object"><freejoint name="task_object_free"/><geom type="sphere" size=".01" mass=".5"/></body></worldbody></mujoco>'
    )
    hands = [
        dict(
            root_body=n,
            fixture_prefix=n + "_",
            q_pregrasp=[0] * 6,
            q_open=[0] * 6,
            q_closed=[0] * 6,
        )
        for n in ("right", "left")
    ]
    return mujoco, model, hands


def source():
    rotation = Rotation.from_euler("z", 0.2).as_matrix()
    rows = [
        dict(
            time_s=t,
            phase="hold" if t >= 4 else "lift",
            object_position_m=[0, 0, z],
            object_rotation=r.tolist(),
        )
        for t, z, r in [
            (0, 0, np.eye(3)),
            (1.8, 0, np.eye(3)),
            (4, 0.1, rotation),
            (5, 0.2, rotation),
        ]
    ]
    return dict(
        configuration=dict(rollout=dict(approach_s=1.0, close_s=0.8, hold_s=14.0)),
        records=rows,
        continuous_protocol=dict(
            duration_s=17.0, acquisition_end_s=1.8, capture_duration_s=12.0
        ),
    )


def test_prefix_rotates_both_wrists_about_object_and_freezes_without_writing_object():
    mujoco, model, hands = fixture_model()
    settled = dict(
        settled_world_from_object=np.eye(4).tolist(), object_qpos=[0, 0, 0, 1, 0, 0, 0]
    )
    cmd = Commands(model, source(), hands, settled, "continuous_prefix_hold")
    initial = cmd.at(0)
    at_hold = cmd.at(4)
    np.testing.assert_allclose(cmd.at(12), at_hold)
    obj = model.joint("task_object_free").qposadr[0]
    np.testing.assert_array_equal(at_hold[obj : obj + 7], initial[obj : obj + 7])
    data = mujoco.MjData(model)
    data.qpos[:] = at_hold
    mujoco.mj_forward(model, data)
    goal = cmd.goal(4)
    for name, x in [("right", -0.2), ("left", 0.2)]:
        np.testing.assert_allclose(
            data.xpos[model.body(name).id],
            goal[:3, :3] @ np.array([x, 0, 0.1]) + goal[:3, 3],
            atol=1e-12,
        )
        np.testing.assert_allclose(
            data.xmat[model.body(name).id].reshape(3, 3), goal[:3, :3], atol=1e-12
        )
    assert cmd.end_close == 1.8 and cmd.hold_start == 4 and cmd.duration == 17


def test_staged_lift_is_explicit_and_does_not_retime_reference_in_place():
    _, model, hands = fixture_model()
    original = source()
    settled = dict(
        settled_world_from_object=np.eye(4).tolist(), object_qpos=[0, 0, 0, 1, 0, 0, 0]
    )
    cmd = Commands(model, original, hands, settled, "staged_hold")
    assert cmd.hold_start == 3.8 and cmd.duration == 17.8
    np.testing.assert_allclose(cmd.goal(1.8), np.eye(4))
    assert cmd.goal(3.8)[2, 3] == pytest.approx(0.07)
    assert original["records"][-1]["object_position_m"][2] == 0.2


def test_acquisition_blend_changes_robot_goal_only_then_rejoins_native_clock():
    from copy import deepcopy

    _, model, hands = fixture_model()
    original = source()
    original["records"][1]["object_position_m"] = [0.004, 0, 0.002]
    before = deepcopy(original)
    settled = dict(
        settled_world_from_object=np.eye(4).tolist(), object_qpos=[0, 0, 0, 1, 0, 0, 0]
    )
    blended = Commands(model, original, hands, settled, "continuous_prefix_hold", 0.4)
    native = Commands(model, original, hands, settled, "continuous_prefix_hold")
    np.testing.assert_allclose(blended.goal(1.8), np.eye(4))
    for t in [2.2, 3.0, 4.0, 12.0]:
        np.testing.assert_allclose(blended.goal(t), native.goal(t), atol=1e-12)
    assert original == before
    with pytest.raises(ValueError, match="half a second"):
        Commands(model, original, hands, settled, "continuous_prefix_hold", 0.6)
