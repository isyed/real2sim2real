from copy import deepcopy

import numpy as np
import pytest

from real2sim2real.controller_actor_training.finger_closure import (
    closure_target,
    FingerSurfaceController,
)


def test_named_closure_preserves_defaults_and_rejects_wrist_or_unknown_joints():
    names = ["fixture_x", "right_index_MCP_FE", "right_index_PIP", "right_thumb_IP"]
    q = np.zeros(4)
    np.testing.assert_array_equal(closure_target(names, q, 0.45), [0, 0.45, 0.45, 0.45])
    np.testing.assert_array_equal(
        closure_target(names, q, 0.45, dict(offsets_rad={"right_index_PIP": -0.1})),
        [0, 0.45, -0.1, 0.45],
    )
    for changes in (
        {"fixture_x": 0.1},
        {"alien_PIP": 0.1},
        {"right_index_PIP": float("nan")},
        {"right_index_PIP": 0.7},
    ):
        with pytest.raises(ValueError):
            closure_target(names, q, 0.45, dict(offsets_rad=changes))
    np.testing.assert_array_equal(q, 0)
    for invalid in (
        {},
        {"offsets_rad": []},
        {"offsets_rad": {"right_index_PIP": [0.1, 0.2]}},
    ):
        with pytest.raises(ValueError):
            closure_target(names, q, 0.45, invalid)


def test_frozen_validation_rejects_closure_and_surface_retuning():
    from real2sim2real.controller_actor_training.object_grasp_pilot import (
        verify_frozen_controller,
    )

    c = dict(
        finger_closure=dict(offsets_rad={"right_index_PIP": 0.1}),
        finger_surface_feedback=dict(gain_s_inv=3),
    )
    selected = dict(controller=deepcopy(c))
    verify_frozen_controller(c, selected)
    for key in c:
        other = deepcopy(c)
        other[key] = {}
        with pytest.raises(ValueError, match="feedback"):
            verify_frozen_controller(other, selected)


def test_solver_reset_cannot_masquerade_as_valid_physics():
    from real2sim2real.controller_actor_training.simulation_health import step_rejection

    assert step_rejection(1.0, 1.002, 0.002, {"BADQACC": 0}) is None
    assert (
        step_rejection(1.0, 0.002, 0.002, {"BADQACC": 1})
        == "simulation_clock_reset_or_invalid_step"
    )
    assert step_rejection(1.0, 1.002, 0.002, {"BADQACC": 1}) == "solver_warning"
    assert step_rejection(1.0, float("nan"), 0.002, {}) == "invalid_simulation_clock"


def test_surface_servo_tracks_with_fingers_only_and_respects_force_travel_limits():
    mj = pytest.importorskip("mujoco")
    from real2sim2real.retargeting.grasp_kinematics import robot_indices

    m = mj.MjModel.from_xml_string("""<mujoco><compiler angle="radian"/>
    <worldbody><body name="hand"><joint name="fixture_x" type="slide" range="-.1 .1"/>
    <geom type="sphere" size=".01"/><site name="thumb"/>
    <body name="index1"><joint name="j1" axis="0 0 1" range="-1 1"/>
    <geom type="capsule" size=".005" fromto="0 0 0 .05 0 0"/>
    <body name="index2" pos=".05 0 0"><joint name="j2" axis="0 0 1" range="-1 1"/>
    <geom type="capsule" size=".005" fromto="0 0 0 .05 0 0"/>
    <body name="index3" pos=".05 0 0"><joint name="j3" axis="0 0 1" range="-1 1"/>
    <geom type="capsule" size=".005" fromto="0 0 0 .05 0 0"/><site name="index" pos=".05 0 0"/>
    </body></body></body></body><body name="task_object"><freejoint/><geom type="sphere" size=".01"/></body></worldbody></mujoco>""")
    d = mj.MjData(m)
    idx = robot_indices(m, "hand")
    from real2sim2real.controller_actor_training.finger_closure import closure_geometry

    original_state = d.qpos.copy()
    audit = closure_geometry(
        m,
        d,
        idx,
        ["thumb", "index"],
        np.zeros(4),
        [0, 0.45, 0.45, 0.45],
        [[0, 0, 0], [0.15, 0, 0]],
    )
    assert max(audit["planned_target_error_m"]) < 1e-12
    assert audit["command_tip_displacement_m"][1] > 0.05
    assert not audit["physical_contact_validated"]
    np.testing.assert_array_equal(d.qpos, original_state)
    s = dict(
        groups=[dict(name="index", site="index", joints=["j1", "j2", "j3"])],
        gain_s_inv=4,
        max_speed_rad_s=0.8,
        max_offset_rad=0.6,
        target_force_n=2,
        force_gain_m_per_n_s=0.001,
        max_inset_m=0.002,
        max_outset_m=0.002,
    )
    config = dict(robot=dict(sites=["thumb", "index"]), finger_surface_feedback=s)
    candidate = dict(
        joint_names=[m.joint(int(j)).name for j in idx["joints"]],
        pair=dict(close_axis=[1, 0, 0]),
        thumb_side_direction=1,
        targets_m=[[0, 0, 0], [0.12, 0.06, 0]],
    )
    c = FingerSurfaceController(m, config, candidate, idx, np.eye(4))
    original = d.qpos.copy()
    q = np.zeros(4)
    first = c.target(m, d, q, {}, 0.01)
    np.testing.assert_array_equal(d.qpos, original)
    assert first[0] == 0 and np.linalg.norm(first[1:]) <= 0.008 + 1e-12
    initial_error = c.groups[0]["error_m"]
    for _ in range(150):
        commands = c.target(m, d, q, {}, 0.01)
        d.qpos[idx["q"]] = commands  # ideal tracking for this kinematic test only
    assert c.groups[0]["error_m"] < initial_error / 2
    assert np.abs(commands).max() <= 0.6 and c.groups[0]["inset"] == 0.002
    np.testing.assert_array_equal(d.qpos[4:], original[4:])
    for _ in range(50):
        c.target(m, d, q, {"index": 20}, 0.01)
    assert c.groups[0]["inset"] == -0.002
    bad = deepcopy(config)
    bad["finger_surface_feedback"]["groups"][0]["joints"] = ["fixture_x"]
    with pytest.raises(ValueError, match="finger ancestors"):
        FingerSurfaceController(m, bad, candidate, idx, np.eye(4))
