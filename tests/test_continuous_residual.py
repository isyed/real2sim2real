import numpy as np
import pytest

from real2sim2real.controller_actor_training.continuous_residual import (
    phase_costs,
    stage_at,
    validate_baseline,
)
from real2sim2real.controller_actor_training.arm_residual_observation import (
    ArmResidualObservation,
)
from real2sim2real.controller_actor_training.residual_pressure import PressureResidual


def protocol():
    return dict(
        schema="real2sim2real.continuous-reference-protocol/1",
        acquisition_end_s=1.8,
        capture_duration_s=12.0,
        duration_s=16.0,
        native_timing_preserved=True,
        raw_reference_preserved=True,
        release_planned=True,
        finger_release=dict(
            start_s=13.0,
            duration_s=1.0,
            settle_s=2.0,
            support_dwell_s=0.2,
            maximum_clearance_m=0.005,
            maximum_speed_m_s=0.02,
        ),
    )


def test_stage_uses_actual_release_latch_and_preserves_unsupported_placement():
    p = protocol()
    assert stage_at(0, "approach", p, None, 0) == "approach"
    assert stage_at(1.4, "close", p, None, 0) == "acquisition"
    assert stage_at(2, "lift", p, None, 0) == "carry"
    assert stage_at(11, "return", p, None, 0) == "carry"
    # Scheduled opening alone must not claim supported release has started.
    assert stage_at(13.5, "release", p, None, 0) == "placement"
    assert stage_at(13.5, "release", p, 13.2, 0.2) == "release"
    assert stage_at(15, "released_settle", p, 13.2, 1) == "settled"


def test_contact_retention_switches_to_unloading_only_after_supported_release():
    args = dict(
        scheduled_phase="hold",
        position_error=np.zeros(3),
        rotation_error=np.zeros(3),
        drift=np.ones(6) * 0.01,
        anchor_valid=True,
        reference_valid=True,
        contact_cost=4,
        clearance_m=0.03,
        minimum_clearance_m=0.05,
        table_load_fraction=0,
        hand_force_n=20,
        release_fraction=0.5,
        open_error_rad=0.1,
    )
    carry = phase_costs({}, stage="carry", **args)
    assert carry["grasp"] > 0 and carry["contact"] == 4
    assert carry["clearance"] == pytest.approx(4) and carry["release"] == 0
    placed = phase_costs({}, stage="placement", **args)
    assert placed["grasp"] == placed["contact"] == placed["release"] == 0
    assert placed["placement"] > 0
    released = phase_costs({}, stage="release", **args)
    assert released["grasp"] == released["contact"] == 0 and released["release"] > 0
    unloaded = phase_costs({}, stage="release", **dict(args, hand_force_n=0))
    assert unloaded["release"] < released["release"]


def test_continuous_anchor_matches_first_post_acquisition_evaluator_sample():
    observer = ArmResidualObservation.__new__(ArmResidualObservation)
    observer.protocol, observer.anchors = protocol(), None
    sample = dict(
        time_s=1.79,
        phase="lift",
        object_position_m=[0.1, 0.2, 0.3],
        wrist_position_m=[0, 0, 0],
        support_wrist_position_m=[0, 0, 0],
        wrist_rotation=np.eye(3),
        support_wrist_rotation=np.eye(3),
    )
    observer.record(sample)
    assert observer.anchors is None
    observer.record(dict(sample, time_s=1.82))
    original = np.array(observer.anchors)
    observer.record(dict(sample, time_s=4, phase="hold", object_position_m=[9, 9, 9]))
    np.testing.assert_array_equal(observer.anchors, original)


def test_continuous_validation_rejects_stationary_shortened_and_retimed_sources():
    baseline = dict(
        continuous_protocol=protocol(),
        motion_condition="demonstrated",
        demonstrated_motion_executed=True,
        physics_timestep_s=0.000125,
        records=[dict(time_s=0.000125), dict(time_s=16.0)],
    )
    validate_baseline(baseline)
    for changes in [
        dict(motion_condition="stationary_hold"),
        dict(records=[dict(time_s=0.000125), dict(time_s=15.0)]),
        dict(continuous_protocol=dict(protocol(), native_timing_preserved=False)),
    ]:
        with pytest.raises(ValueError):
            validate_baseline(dict(baseline, **changes))


def test_pressure_requests_decay_on_release_and_cannot_enable_release_actuation():
    settings = dict(
        policy_period_s=0.02,
        max_offset_n=1,
        max_rate_n_s=5,
        active_phases=["close", "lift", "hold", "return"],
    )
    control = PressureResidual(settings)
    control.accept(np.ones(4))
    for _ in range(200):
        control.advance(0.001, "return")
    assert control.offset.min() > 0.99
    for _ in range(201):
        control.advance(0.001, "release")
    assert not control.offset.any()
    with pytest.raises(ValueError, match="release"):
        PressureResidual(dict(settings, active_phases=["release"]))
