from copy import deepcopy

import numpy as np
import pytest

from real2sim2real.controller_actor_training.acquisition_rollout_search import (
    evaluate,
    rank,
    read,
    validate_names,
)


def fixture():
    times = np.arange(0, 2.01, 0.02)
    labels = [
        r + ":" + g
        for r in ("primary", "secondary")
        for g in ("thumb", "index", "middle", "ring", "pinky")
    ]
    report = dict(
        schema="real2sim2real.arm-replay-pilot/1",
        physics_run=True,
        motion_condition="stationary_hold",
        object_state_written_only_at_initialization=True,
        config=dict(
            output="old", motion_condition="stationary_hold", kp_nm_per_rad=1200
        ),
        scene_sha256="scene",
        physics_timestep_s=0.000125,
        controller_period_s=0.001,
        continuous_protocol=dict(
            frame_ids=[0, 1],
            capture_duration_s=1.5,
            capture_clock_origin_s=0,
            acquisition_end_s=0.4,
            reference_start_s=0,
        ),
        continuous_metrics=dict(retention_anchor_time_s=0.4),
        hold_slip_m=0.008,
        support_wrist_drift_m=0.009,
        hold_minimum_clearance_m=0.06,
        hold_max_rotation_deg=1,
        max_robot_penetration_m=0.001,
        max_arm_penetration_m=0,
        minimum_dynamic_joint_margin_rad=0.1,
        maximum_dynamic_velocity_ratio=0.5,
        success=True,
        rejected_for=[],
        records=[
            dict(
                time_s=float(t),
                phase="close" if t < 0.4 else "hold",
                reference_position_error_m=0.001,
                reference_rotation_error_deg=1,
                contact_patch_motion=dict(pads={k: dict(force_n=2) for k in labels}),
                contact_force_feedback={k: dict(target_n=4) for k in labels},
            )
            for t in times
        ],
    )
    limits = dict(
        primary_drift_m=0.01,
        support_drift_m=0.01,
        minimum_clearance_m=0.05,
        maximum_reference_position_m=0.02,
        maximum_reference_rotation_deg=10,
        maximum_hold_rotation_deg=30,
        maximum_robot_penetration_m=0.003,
        maximum_arm_penetration_m=0.001,
        minimum_joint_margin_rad=0.03,
        maximum_arm_velocity_ratio=1,
    )
    return report, times, limits


def test_better_contact_score_cannot_promote_a_rejected_controller():
    source, times, limits = fixture()
    failed = deepcopy(source)
    failed.update(success=False, rejected_for=["hold_slip"], hold_slip_m=0.012)
    score = evaluate(failed, source, times, times, limits)
    assert score["complete"] and not score["passed"]
    assert score["sustained_hold_load_fraction"] == 1
    assert score["normalized_metrics"]["primary_drift"] == 1.2
    passed = evaluate(source, source, times, times, limits)
    assert rank(dict(name="pass", evaluation=passed)) < rank(
        dict(name="fail", evaluation=score)
    )
    assert failed["rejected_for"] == ["hold_slip"]


def test_shortened_or_phase_changed_rollout_never_wins_even_if_it_claims_success():
    source, times, limits = fixture()
    shortened = deepcopy(source)
    shortened["records"] = shortened["records"][:30]
    score = evaluate(shortened, source, times[:30], times, limits)
    assert not score["eligible"] and not score["passed"] and score["objective"] is None
    changed = deepcopy(source)
    changed["records"][25]["phase"] = "close"
    assert not evaluate(changed, source, times, times, limits)["eligible"]


@pytest.mark.parametrize(
    "change", ["clock", "physics", "budget", "anchor", "unsafe_success"]
)
def test_search_cannot_change_limits_or_hide_unsafe_success(change):
    source, times, limits = fixture()
    changed = deepcopy(source)
    if change == "clock":
        changed["continuous_protocol"]["capture_duration_s"] = 1
    if change == "physics":
        changed["physics_timestep_s"] = 0.00025
    if change == "budget":
        changed["config"]["kp_nm_per_rad"] = 2000
    if change == "anchor":
        changed["continuous_metrics"]["retention_anchor_time_s"] = 0.6
    if change == "unsafe_success":
        changed["max_robot_penetration_m"] = 0.004
    with pytest.raises(ValueError):
        evaluate(changed, source, times, times, limits)


def test_full_hold_contact_and_rotation_enter_objective_without_changing_gates():
    source, times, limits = fixture()
    missing = deepcopy(source)
    missing.update(success=False, rejected_for=["missing_opposed_finger_contact"])
    for row in missing["records"]:
        row["contact_patch_motion"]["pads"]["secondary:thumb"]["force_n"] = 0
    score = evaluate(missing, source, times, times, limits)
    assert score["sustained_hold_load_fraction"] == 0
    assert score["objective"] > 0.25 and not score["passed"]
    empty = dict(name="empty", evaluation=dict(eligible=False))
    assert rank(dict(name="finished", evaluation=score)) < rank(empty)


def test_changed_input_pin_is_rejected_before_use(tmp_path):
    from real2sim2real.artifacts import digest

    p = tmp_path / "input.json"
    p.write_text("{}")
    spec = dict(path=str(p), sha256=digest(p))
    assert read(spec)[1] == {}
    p.write_text('{"changed": true}')
    with pytest.raises(ValueError, match="changed"):
        read(spec)


@pytest.mark.parametrize(
    "names", [["search"], ["progress"], ["../outside"], ["case", "case-repeat"]]
)
def test_candidate_names_cannot_overwrite_protocols_or_reserved_repeat_outputs(names):
    with pytest.raises(ValueError, match="overwrite"):
        validate_names([dict(name=n) for n in names])
    validate_names([dict(name="thumb-first"), dict(name="hand-first")])
