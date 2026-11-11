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


def test_phase_deviation_reveals_insertion_error_and_never_time_aligns():
    from real2sim2real.controller_actor_training.arm_replay_comparison import (
        phase_object_deviation,
    )
    from copy import deepcopy

    a = [
        dict(phase=p, time_s=t, object_position_m=[0, 0, 0])
        for t, p in enumerate(["lift", "support_approach", "hold"])
    ]
    b = deepcopy(a)
    b[1]["object_position_m"] = [0.04, 0, 0]
    b[2]["object_position_m"] = [0.04, 0, 0]
    result = phase_object_deviation(a, b)
    assert result["by_phase"]["lift"]["maximum_translation_difference_m"] == 0
    assert (
        result["by_phase"]["support_approach"]["maximum_translation_difference_m"]
        == 0.04
    )
    assert not result["temporal_realignment"]
    assert not phase_object_deviation(a, b[:-1])["available"]
    b[1]["time_s"] += 0.01
    assert not phase_object_deviation(a, b)["available"]


def test_comparison_exposes_feedback_rate_changes_at_same_physics_step(tmp_path):
    import json
    from real2sim2real.artifacts import digest
    from real2sim2real.controller_actor_training.arm_replay_comparison import compare

    paths = []
    for name, period in [("legacy", None), ("fixed", 0.001)]:
        folder = tmp_path / name
        folder.mkdir()
        (folder / "scene.xml").write_text(
            '<mujoco><option timestep="0.00025"/></mujoco>'
        )
        (folder / "states.npz").write_bytes(b"preserved states")
        r = report()
        r.update(
            schema="real2sim2real.arm-replay-pilot/1",
            config=dict(audit_sha256="same"),
            scene_sha256=digest(folder / "scene.xml"),
            states_sha256=digest(folder / "states.npz"),
        )
        if period is not None:
            r["controller_period_s"] = period
            r["config"]["control_period_s"] = period
        path = folder / "result.json"
        path.write_text(json.dumps(r))
        paths.append(path)
    out = tmp_path / "comparison"
    r = compare(paths, out)
    assert r["differing_base_control_settings"] == ["control_period_s"]
    assert [row["controller_period_s"] for row in r["rows"]] == [0.00025, 0.001]
    text = (out / "review.html").read_text()
    assert "0.250 / 0.250" in text and "0.250 / 1.000" in text
