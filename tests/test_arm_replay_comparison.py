from real2sim2real.controller_actor_training.arm_replay_comparison import summarize


def report():
    return dict(
        success=False,
        rejected_for=["reference_fidelity"],
        hold_slip_m=0.006,
        support_wrist_drift_m=0.008,
        hold_minimum_clearance_m=0.06,
        hold_opposed_contact_fraction=0.9,
        support_load_fraction=0.9,
        max_robot_penetration_m=0.001,
        max_arm_penetration_m=0,
        maximum_dynamic_velocity_ratio=0.5,
        minimum_dynamic_joint_margin_rad=0.9,
        saturation_fraction={},
        records=[dict(phase="hold")],
    )


def test_missing_reference_is_not_reported_as_zero_error_or_success():
    summary = summarize(report())
    assert summary["maximum_reference_position_error_m"] is None
    assert summary["maximum_reference_rotation_error_deg"] is None
    assert not summary["success"]
    assert summary["rejected_for"] == ["reference_fidelity"]


def test_hold_maximum_and_load_engagement_are_preserved():
    r = report()
    r["records"] = [
        dict(
            phase="approach",
            reference_position_error_m=1,
            reference_rotation_error_deg=90,
        ),
        dict(
            phase="hold",
            reference_position_error_m=0.03,
            reference_rotation_error_deg=5,
            object_feedback=dict(support_load_offset_m=0.0005),
        ),
        dict(
            phase="hold",
            reference_position_error_m=0.01,
            reference_rotation_error_deg=2,
            object_feedback=dict(support_load_offset_m=0),
        ),
    ]
    summary = summarize(r)
    assert summary["maximum_reference_position_error_m"] == 0.03
    assert summary["maximum_reference_rotation_error_deg"] == 5
    assert summary["maximum_support_load_offset_m"] == 0.0005
