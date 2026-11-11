from real2sim2real.controller_actor_training.arm_replay_review import motion_explanation


def test_stationary_banner_explains_command_stop_and_full_simulated_duration():
    report = dict(
        motion_condition="stationary_hold",
        records=[
            dict(time_s=0, phase="approach"),
            dict(time_s=4.02, phase="hold"),
            dict(time_s=16.744375, phase="hold"),
        ],
    )
    text = motion_explanation(report)
    assert "4.02 s" in text and "16.74 s" in text
    assert "deliberately stops" in text and "free to move or slip" in text
    assert "not the full demonstrated trajectory" in text


def test_moving_banner_does_not_claim_stationary_hold_or_success():
    text = motion_explanation(dict(motion_condition="demonstrated", success=False))
    assert "demonstrated-motion condition" in text
    assert "deliberately stops" not in text
    assert "Check the result for task acceptance" in text


def test_legacy_stationary_report_without_records_is_still_labeled():
    assert "saved stationary test" in motion_explanation(
        dict(motion_condition="stationary_hold")
    )


def test_refresh_preserves_video_and_rejects_stale_result_pair(tmp_path, monkeypatch):
    import json
    import pytest
    from real2sim2real.controller_actor_training import arm_replay_review as review
    from real2sim2real.artifacts import digest

    keys = [
        "success",
        "rejected_for",
        "hold_minimum_clearance_m",
        "hold_slip_m",
        "support_wrist_drift_m",
        "hold_opposed_contact_fraction",
        "support_load_fraction",
        "max_robot_penetration_m",
        "max_arm_penetration_m",
        "maximum_joint_tracking_error_rad",
    ]
    report = {key: False for key in keys}
    report["motion_condition"] = "stationary_hold"
    path = tmp_path / "result.json"
    path.write_text(json.dumps(report))
    video = tmp_path / "rollout.mp4"
    video.write_bytes(b"immutable saved video")
    monkeypatch.setattr(review, "prepare_playback", lambda *args: {})
    monkeypatch.setattr(review, "video_html", lambda *args, **kwargs: "SAVED_PLAYER")
    before = digest(video)
    review.refresh_page(tmp_path)
    assert digest(video) == before
    assert "Stationary diagnostic" in (tmp_path / "review.html").read_text()
    path.write_text(json.dumps({**report, "success": True}))
    with pytest.raises(ValueError, match="identity"):
        review.refresh_page(tmp_path)


def test_render_does_not_overwrite_existing_media_before_optional_imports(tmp_path):
    import pytest
    from real2sim2real.controller_actor_training.arm_replay_review import render

    path = tmp_path / "rollout.mp4"
    path.write_bytes(b"original")
    with pytest.raises(FileExistsError, match="refresh-page"):
        render(tmp_path)
    assert path.read_bytes() == b"original"
