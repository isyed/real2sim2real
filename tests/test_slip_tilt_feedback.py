import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from real2sim2real.controller_actor_training.slip_tilt_feedback import SlipTiltFeedback


def settings(**changes):
    return dict(
        slip_gain_s_inv=1.5,
        tilt_gain=0.5,
        height_gain=0,
        filter_gain_s_inv=3,
        max_offset_m=0.006,
        max_angle_rad=0.06,
        max_speed_m_s=0.012,
        max_angular_speed_rad_s=0.12,
        minimum_contact_n=1,
        maximum_reference_error_m=0.02,
        maximum_reference_angle_rad=np.deg2rad(10),
        observation_period_s=0.02,
        **changes,
    )


def make(**changes):
    s = settings()
    s.update(changes)
    p = dict(acquisition_end_s=1, capture_duration_s=2, finger_release=dict(start_s=3))
    c = SlipTiltFeedback(s, p)
    sample = dict(
        time_s=1,
        object_position_m=[0, 0, 0],
        wrist_position_m=[-0.1, 0, 0],
        support_wrist_position_m=[0.1, 0, 0],
        wrist_rotation=np.eye(3),
        support_wrist_rotation=np.eye(3),
    )
    c.record(sample)
    return c, sample


def poses():
    actual, reference = np.eye(4), np.eye(4)
    wrists = [np.eye(4), np.eye(4)]
    wrists[0][0, 3] = -0.1
    wrists[1][0, 3] = 0.1
    return actual, reference, wrists


def test_fixed_anchor_slip_direction_travel_rate_and_contact_freeze():
    c, sample = make(tilt_gain=0)
    anchors = c.anchors.copy()
    c.record(dict(sample, time_s=1.1, object_position_m=[9, 9, 9]))
    np.testing.assert_array_equal(c.anchors, anchors)
    actual, ref, wrists = poses()
    actual[0, 3] = 0.004
    old = np.zeros((2, 6))
    for _ in range(500):
        new = c.update(1.5, 0.001, actual, ref, wrists, wrists, [2, 2])
        assert (
            np.max(np.linalg.norm(new[:, :3] - old[:, :3], axis=1))
            <= 0.012 * 0.001 + 1e-12
        )
        assert np.max(np.linalg.norm(new[:, :3], axis=1)) <= 0.006 + 1e-12
        old = new
    assert (new[:, 0] > 0).all()
    np.testing.assert_array_equal(
        c.update(1.6, 0.001, actual, ref, wrists, wrists, [0, 0]), new
    )
    actual[0, 3] = 0.03
    np.testing.assert_array_equal(
        c.update(1.6, 0.001, actual, ref, wrists, wrists, [2, 2]), new
    )


def test_tilt_uses_coordinated_object_pivot_and_requires_both_contacts():
    c, _ = make(slip_gain_s_inv=0)
    actual, ref, wrists = poses()
    actual[:3, :3] = Rotation.from_euler("z", 0.05).as_matrix()
    assert not c.update(1.5, 0.001, actual, ref, wrists, wrists, [2, 0]).any()
    for _ in range(500):
        new = c.update(1.5, 0.001, actual, ref, wrists, wrists, [2, 2])
    assert (new[:, 5] < 0).all()
    assert new[0, 1] > 0 and new[1, 1] < 0
    assert new[0, 5] == pytest.approx(new[1, 5])


def test_downward_slip_is_blocked_and_height_feedback_only_lifts():
    c, _ = make(tilt_gain=0)
    actual, ref, wrists = poses()
    actual[2, 3] = -0.004
    for _ in range(100):
        result = c.update(1.5, 0.001, actual, ref, wrists, wrists, [2, 2])
    assert not result.any() and c.blocked_downward > 0
    c, _ = make(tilt_gain=0, height_gain=0.5)
    assert (c.update(1.5, 0.001, actual, ref, wrists, wrists, [2, 2])[:, 2] > 0).all()


def test_release_recovers_even_on_contact_loss_and_stationary_retains_offsets():
    c, _ = make(tilt_gain=0)
    actual, ref, wrists = poses()
    actual[0, 3] = 0.004
    for _ in range(500):
        initial = c.update(1.5, 0.001, actual, ref, wrists, wrists, [2, 2])
    np.testing.assert_array_equal(
        c.update(2, 0.001, actual, ref, wrists, wrists, [0, 0]), initial
    )
    np.testing.assert_allclose(
        c.update(2.5, 0.001, actual, ref, wrists, wrists, [0, 0]), initial * 0.5
    )
    assert not c.update(3, 0.001, actual, ref, wrists, wrists, [0, 0]).any()
    c, _ = make(tilt_gain=0)
    c.protocol["release_disabled_for_stationary_diagnostic"] = True
    for _ in range(100):
        initial = c.update(1.5, 0.001, actual, ref, wrists, wrists, [2, 2])
    np.testing.assert_array_equal(
        c.update(3.5, 0.001, actual, ref, wrists, wrists, [0, 0]), initial
    )


def test_anchor_validity_zero_gains_and_invalid_budgets():
    c, _ = make(slip_gain_s_inv=0, tilt_gain=0)
    actual, ref, wrists = poses()
    actual[0, 3] = 0.004
    assert not c.update(1.5, 0.001, actual, ref, wrists, wrists, [2, 2]).any()
    c.anchors = None
    assert not c.update(0.5, 0.001, actual, ref, wrists, wrists, [2, 2]).any()
    for change in [
        dict(max_offset_m=0.007),
        dict(observation_period_s=0.001),
        dict(max_speed_m_s=0.001),
        dict(slip_gain_s_inv=np.nan),
    ]:
        with pytest.raises(ValueError):
            make(**change)


def test_command_collision_guard_checks_midpoint_without_live_state_writes():
    mj = pytest.importorskip("mujoco")
    from real2sim2real.controller_actor_training.closed_loop_grasp import (
        ClosedLoopGrasp,
    )

    m = mj.MjModel.from_xml_string("""<mujoco><compiler autolimits="true"/><worldbody>
    <geom name="obstacle" type="sphere" size=".015"/>
    <body name="robot"><joint name="slide" type="slide" axis="1 0 0" range="-.2 .2"/>
    <geom name="hand" type="sphere" size=".015" mass="1"/></body>
    <body name="task_object" pos="5 0 0"><freejoint/><geom name="obj" size=".02" mass="1"/></body>
    </worldbody></mujoco>""")
    d = mj.MjData(m)
    d.qpos[0] = -0.1
    mj.mj_forward(m, d)
    c = ClosedLoopGrasp.__new__(ClosedLoopGrasp)
    c.model, c.live, c.guard = m, d, mj.MjData(m)
    c.qids = np.array([0])
    c.robot_geoms = {m.geom("hand").id}
    c.arm_geoms = set()
    c.object_geoms = {m.geom("obj").id}
    c.criteria = dict(arm_penetration_m=0.001, robot_penetration_m=0.003)
    c.ever_corrected = True
    c.previous_checked = np.array([-0.1])
    c.guard_checks = c.guard_rejections = 0
    before = {
        k: getattr(d, k).copy()
        for k in ["qpos", "qvel", "ctrl", "qacc", "qacc_warmstart"]
    }
    # Both endpoints are clear; the midpoint crosses the obstacle.
    assert not c.check_command(np.array([0.1]))
    assert c.guard_rejections == 1 and c.worst_contact["penetration_m"] > 0.003
    assert c.check_command(np.array([-0.12]))
    for k, v in before.items():
        np.testing.assert_array_equal(getattr(d, k), v)
