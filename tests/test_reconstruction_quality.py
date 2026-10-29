import numpy as np
import pytest
from real2sim2real.reconstruction.quality import QualityLimits, quality_report


def test_tracker_success_does_not_hide_image_and_motion_failure():
    poses = np.tile(np.eye(4), (4, 1, 1))
    poses[2:, :3, 3] = [0.2, 0, 0]
    report = quality_report(
        poses,
        np.arange(4) / 30,
        image_iou=[0.8, 0.1, 0.2, 0.9],
        tracker_reliable=np.ones(4, bool),
    )
    assert report["tracker_reliable_frames"] == 4
    assert report["image_agreement"]["low_iou_frames"] == [1, 2]
    assert report["continuity"]["translation_warning_frames"] == [2]
    assert report["status"] == "review_required"
    assert (
        not report["approved_for_actor_training"] and not report["predictions_modified"]
    )


def test_good_scores_are_not_physical_approval_and_missing_scores_are_explicit():
    poses = np.tile(np.eye(4), (2, 1, 1))
    good = quality_report(poses, [0, 0.1], image_iou=[0.9, 0.9])
    assert good["status"] == "no_configured_warnings"
    assert (
        not good["approved_for_retargeting"]
        and "support_placement" in good["unassessed"]
    )
    missing = quality_report(poses, [0, 0.1])
    assert missing["image_agreement"]["mean_iou"] is None
    assert missing["image_agreement"]["missing_frames"] == [0, 1]
    assert missing["status"] == "review_required"


def test_invalid_timeline_pose_scores_and_limits_fail():
    poses = np.tile(np.eye(4), (2, 1, 1))
    for times in ([0, 0], [0], [0, float("nan")]):
        with pytest.raises(ValueError):
            quality_report(poses, times)
    with pytest.raises(ValueError):
        quality_report(poses, [0, 1], image_iou=[2, 0.5])
    with pytest.raises(ValueError):
        quality_report(poses, [0, 1], limits=QualityLimits(max_translation_step_m=-1))
    poses[0, 0, 0] = -1
    with pytest.raises(ValueError):
        quality_report(poses, [0, 1])
