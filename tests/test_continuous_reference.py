from copy import deepcopy

import numpy as np
import pytest

from real2sim2real.controller_actor_training.continuous_reference import (
    reference_goal,
    table_command,
    phase_at,
    evaluate_continuous,
)
from real2sim2real.controller_actor_training.guided_motion import MotionReference
from real2sim2real.controller_actor_training.arm_motion_condition import apply_condition


def test_capture_clock_has_no_independent_lift_or_reference_restart():
    poses = np.repeat(np.eye(4)[None], 101, axis=0)
    times = np.linspace(0, 5, len(poses))
    poses[:, 2, 3] = np.maximum(times - 3, 0) * 0.05
    ref = MotionReference(times, poses, np.arange(len(poses)))
    anchor = np.eye(4)
    anchor[:3, 3] = [1, 2, 0.04]
    np.testing.assert_array_equal(reference_goal(ref, 2, anchor), anchor)
    assert reference_goal(ref, 4, anchor)[2, 3] == pytest.approx(0.09)
    np.testing.assert_array_equal(ref.poses, poses)


def test_table_projection_preserves_raw_reference_and_rejects_excess():
    goal = np.eye(4)
    goal[2, 3] = -0.006
    original = goal.copy()
    command, clearance, adjustment = table_command(goal, np.zeros((1, 3)), 0, 0.01)
    assert clearance == -0.006 and adjustment == 0.006
    assert command[2, 3] == 0
    np.testing.assert_array_equal(goal, original)
    with pytest.raises(ValueError, match="excessive"):
        table_command(goal, np.zeros((1, 3)), 0, 0.005)
    goal[2, 3] = float("nan")
    with pytest.raises(ValueError, match="Finite"):
        table_command(goal, np.zeros((1, 3)), 0, 0.01)


def test_phases_use_independent_reference_clearance_and_capture_end():
    assert phase_at(0.5, 1, 0.8, 0, 0.05, False, 12) == "approach"
    assert phase_at(1.5, 1, 0.8, 0, 0.05, False, 12) == "close"
    assert phase_at(2, 1, 0.8, 0, 0.05, False, 12) == "lift"
    assert phase_at(5, 1, 0.8, 0.1, 0.05, False, 12) == "hold"
    assert phase_at(9, 1, 0.8, 0.01, 0.05, True, 12) == "return"
    assert phase_at(12.5, 1, 0.8, 0, 0.05, True, 12) == "settle"


def records():
    return [
        dict(
            time_s=t,
            phase=p,
            object_position_m=[x, 0, 0],
            object_rotation=np.eye(3).tolist(),
            wrist_position_m=[0, 0, 0],
            wrist_rotation=np.eye(3).tolist(),
            support_wrist_position_m=[0, 0, 0],
            support_wrist_rotation=np.eye(3).tolist(),
            minimum_object_height_m=0,
            table_contacts=1,
            reference_position_m=[0, 0, 0],
            reference_rotation=np.eye(3).tolist(),
        )
        for t, p, x in [
            (1.8, "lift", 0),
            (4, "hold", 0.011),
            (8, "hold", 0.011),
            (12, "settle", 0.011),
            (13, "settle", 0.011),
        ]
    ]


def test_retention_is_anchored_at_acquisition_not_reset_after_lift():
    protocol = dict(acquisition_end_s=1.8, capture_duration_s=12, terminal_settle_s=1)
    rows = records()
    metrics, failures = evaluate_continuous(rows, protocol, 0)
    assert not failures
    assert metrics["hold_slip_m"] == 0.011
    assert metrics["support_wrist_drift_m"] == 0.011
    assert metrics["retention_anchor_time_s"] == 1.8
    rows[-1]["minimum_object_height_m"] = -0.004
    assert "object_table_penetration" in evaluate_continuous(rows, protocol, 0)[1]
    rows = records()
    rows[-1]["table_contacts"] = 0
    assert "incomplete_table_return" in evaluate_continuous(rows, protocol, 0)[1]
    assert not evaluate_continuous(rows, protocol, 0, stationary=True)[1]


def test_stationary_diagnostic_freezes_return_without_changing_source():
    parent = dict(records=records(), continuous_protocol={"schema": "test"})
    original = deepcopy(parent)
    targets = np.arange(10).reshape(5, 2)
    actual, held = apply_condition(parent, targets, "stationary_hold")
    assert parent == original
    assert all(r["phase"] == "hold" for r in actual["records"][1:])
    np.testing.assert_array_equal(held[1:], np.tile(targets[1], (4, 1)))


def test_release_does_not_hide_capture_drift_and_requires_actual_open_contacts():
    protocol = dict(
        acquisition_end_s=1.8,
        capture_duration_s=12,
        terminal_settle_s=1,
        release_planned=True,
        finger_release=dict(start_s=13.1, duration_s=0.8, settle_s=0.8),
    )
    rows = records()
    for t in [13.91, 14.3, 14.69]:
        r = deepcopy(rows[-1])
        r.update(
            time_s=t,
            phase="released_settle",
            wrist_position_m=[1, 0, 0],
            finger_release=dict(fraction=1.0, maximum_open_error_rad=0.01),
            contact_patch_motion=dict(
                pads={"test": {"force_n": 0.0}}, other_object_contacts_n={}
            ),
        )
        rows.append(r)
    metrics, failures = evaluate_continuous(rows, protocol, 0)
    assert not failures and metrics["release"]["accepted"]
    assert metrics["hold_slip_m"] == 0.011
    rows[-1]["finger_release"]["maximum_open_error_rad"] = 0.2
    assert "incomplete_finger_release" in evaluate_continuous(rows, protocol, 0)[1]
    parent = dict(
        records=rows,
        continuous_protocol=protocol,
        configuration=dict(rollout=dict(finger_release=protocol["finger_release"])),
    )
    stationary, _ = apply_condition(parent, np.zeros((len(rows), 2)), "stationary_hold")
    assert "finger_release" not in stationary["configuration"]["rollout"]
    assert parent["configuration"]["rollout"]["finger_release"]


def test_incomplete_release_keeps_a_serializable_rejection():
    import json

    protocol = dict(
        acquisition_end_s=1.8,
        capture_duration_s=12,
        terminal_settle_s=1,
        finger_release=dict(start_s=13.1, duration_s=0.8, settle_s=0.8),
    )
    metrics, failures = evaluate_continuous(records()[:3], protocol, 0)
    assert "incomplete_table_return" in failures
    assert "incomplete_finger_release" in failures
    json.dumps(metrics, allow_nan=False)


def test_per_hand_force_diagnostics_keep_the_combined_release_limit():
    protocol = dict(
        acquisition_end_s=1.8,
        capture_duration_s=12,
        terminal_settle_s=1,
        finger_release=dict(start_s=13.1, duration_s=0.8, settle_s=0.8),
    )
    rows = records()
    for t in [13.91, 14.3, 14.69]:
        r = deepcopy(rows[-1])
        r.update(
            time_s=t,
            finger_release=dict(fraction=1.0, maximum_open_error_rad=0.01),
            contact_patch_motion=dict(
                pads={
                    "primary:index": {"force_n": 0.4},
                    "secondary:pinky": {"force_n": 0.6},
                },
                other_object_contacts_n={"secondary:pinky_MP": 0.3},
            ),
        )
        rows.append(r)
    result, failures = evaluate_continuous(rows, protocol, 0)
    assert result["release"]["peak_hand_forces_n"] == pytest.approx(
        dict(primary=0.4, secondary=0.9)
    )
    assert result["release"]["maximum_hand_object_force_n"] == pytest.approx(1.3)
    assert "incomplete_finger_release" in failures
