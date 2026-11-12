from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from real2sim2real.controller_actor_training.acquisition_profile import (
    identity,
    validate,
    warp,
    arm_targets,
    bind_fingers,
    screen,
)
from real2sim2real.controller_actor_training.arm_finger_feedback import finger_waypoint


def test_identity_preserves_exact_motor_commands_and_acquisition_anchor():
    times = np.linspace(0, 3, 151)
    commands = np.column_stack([np.sin(times), np.cos(times)])
    arms = [dict(name="right", joints=["r"]), dict(name="left", joints=["l"])]
    rollout = dict(approach_s=1, close_s=0.8)
    np.testing.assert_array_equal(
        arm_targets(times, commands, arms, rollout, identity()), commands
    )
    for time in [0, 1, 1.3, 1.8, 2.1]:
        expected = finger_waypoint(time, [0], [0.2], [0.6], rollout)
        actual = finger_waypoint(
            time, [0], [0.2], [0.6], rollout, closing_windows=[[0, 1]]
        )
        np.testing.assert_array_equal(expected, actual)
    np.testing.assert_array_equal(warp(times, 1, 0.8, [0, 1]), times)


def test_sequencing_keeps_open_closed_targets_and_never_changes_reference_clock():
    rollout = dict(approach_s=1, close_s=0.8)
    windows = [[0, 0.5], [0.5, 1]]
    np.testing.assert_allclose(
        finger_waypoint(1.4, [0, 0], [0, 0], [1, 1], rollout, closing_windows=windows),
        [1, 0],
    )
    np.testing.assert_array_equal(
        finger_waypoint(1.8, [0, 0], [0, 0], [1, 1], rollout, closing_windows=windows),
        [1, 1],
    )
    original = deepcopy(rollout)
    np.testing.assert_allclose(
        warp([0, 1, 1.4, 1.8, 8], 1, 0.8, [0, 0.5]), [0, 1, 1.8, 1.8, 8]
    )
    assert rollout == original
    with pytest.raises(ValueError):
        finger_waypoint(
            1.4,
            [0],
            [0],
            [1],
            dict(rollout, finger_close_fraction=0.5),
            closing_windows=[[0, 0.5]],
        )


def test_binding_cannot_take_arm_or_object_ownership_or_mutate_source_profile():
    live = SimpleNamespace(
        rollout=dict(approach_s=1, close_s=0.8),
        hands=[dict(qids=[0, 1])],
        servo=SimpleNamespace(
            pads=[
                dict(role="primary", group="thumb", qids=[0], dofs=[0]),
                dict(role="primary", group="index", qids=[1], dofs=[1]),
            ]
        ),
    )
    model = SimpleNamespace(
        dof_jntid=np.array([0, 1]),
        joint=lambda i: SimpleNamespace(name=["thumb", "index"][i]),
    )
    profile = identity()
    profile["finger_windows"]["primary"]["thumb"] = [0, 0.5]
    original = deepcopy(profile)
    bind_fingers(live, model, profile)
    assert profile == original
    np.testing.assert_array_equal(live.hands[0]["closing_windows"], [[0, 0.5], [0, 1]])
    profile["joint_windows"] = dict(arm_joint=[0, 0.5])
    before = live.hands[0]["closing_windows"].copy()
    with pytest.raises(ValueError, match="owned|declared"):
        bind_fingers(live, model, profile)
    np.testing.assert_array_equal(live.hands[0]["closing_windows"], before)


def test_retimed_arm_speed_and_margin_need_fresh_admission():
    model = SimpleNamespace(
        jnt_qposadr=np.array([0, 1]), jnt_range=np.array([[-1, 1], [-1, 1]])
    )
    t = [0, 0.1]
    assert screen(model, [0, 1], np.array([[0, 0], [0.1, 0.1]]), t, [2, 2], 0.03)[
        "accepted"
    ]
    assert not screen(model, [0, 1], np.array([[0, 0], [0.4, 0.1]]), t, [2, 2], 0.03)[
        "accepted"
    ]
    assert not screen(
        model, [0, 1], np.array([[0, 0], [0.99, 0.1]]), t, [20, 20], 0.03
    )["accepted"]


@pytest.mark.parametrize(
    "value", [[0, 0.3], [0.5, 0.6], [-0.1, 1], [0, 1.1], [0, float("nan")]]
)
def test_profiles_cannot_move_closure_outside_original_interval(value):
    p = identity()
    p["wrist_windows"]["right"] = value
    with pytest.raises(ValueError):
        validate(p)


def test_pressure_profile_preserves_budgets_and_rejects_conflicting_seek_before_binding():
    from test_contact_force_control import settings

    model = SimpleNamespace(
        dof_jntid=np.array([0]), joint=lambda i: SimpleNamespace(name="thumb")
    )
    force = settings()
    force["unloaded_acquisition"] = dict(
        start_fraction=0.6,
        end_after_close_s=0.6,
        gain_m_per_n_s=0.003,
        max_normal_speed_m_s=0.006,
    )
    live = SimpleNamespace(
        rollout=dict(approach_s=1, close_s=0.8),
        settings=force,
        hands=[dict(qids=[0])],
        servo=SimpleNamespace(
            pads=[dict(role="primary", group="thumb", qids=[0], dofs=[0])]
        ),
    )
    profile = identity()
    profile["pressure_activation_fraction"] = 0.5
    profile["finger_windows"]["primary"]["thumb"] = [0, 0.5]
    original = deepcopy(force)
    with pytest.raises(ValueError, match="acquisition"):
        bind_fingers(live, model, profile)
    assert force == original and "closing_windows" not in live.hands[0]
    force.pop("unloaded_acquisition")
    original = deepcopy(force)
    bind_fingers(live, model, profile)
    assert {k: v for k, v in force.items() if k != "close_activation_fraction"} == {
        k: v for k, v in original.items() if k != "close_activation_fraction"
    }
    assert force["close_activation_fraction"] == 0.5
    with pytest.raises(ValueError, match="fresh"):
        bind_fingers(live, model, identity())
