import numpy as np
import pytest

from real2sim2real.egocentric.hand_point_tracks import (
    select_seeds,
    aggregate_tracks,
    confidence,
)
from real2sim2real.egocentric.mano_refinement import image_evidence
from real2sim2real.egocentric.point_hand_experiment import runtime_config
from real2sim2real.egocentric.point_hand_review import boundary_diagnostics


def test_boundary_check_detects_finger_jump_with_stationary_wrist():
    joints = np.zeros((3, 2, 21, 3))
    joints[1, 0, 12, 0] = 0.1
    baseline = np.ones((3, 2), bool)
    baseline[1, 0] = False
    supported = np.ones_like(baseline)
    rows = boundary_diagnostics(
        joints, baseline, supported, [40, 41, 42], [0, 0.05, 0.1]
    )
    assert len(rows) == 2
    assert all(r["wrist_step_m"] == 0 and r["worst_joint"] == 12 for r in rows)
    assert all(r["max_joint_speed_m_s"] == pytest.approx(2) for r in rows)
    supported[1, 0] = False
    assert not boundary_diagnostics(
        joints, baseline, supported, [40, 41, 42], [0, 0.05, 0.1]
    )


def fixture_tracks():
    masks = np.zeros((4, 2, 30, 40), bool)
    masks[:, 0, 5:25, 5:25] = True
    masks[:, 1, 5:25, 30:40] = True
    seeds = dict(
        query_tyx=np.array([[0, 10, 10], [3, 10, 10]]),
        hand=np.array([0, 0]),
        joint=np.array([4, 4]),
    )
    tracks = np.full((2, 4, 2), 10.0)
    scores = np.full((2, 4), 0.95)
    return masks, seeds, tracks, scores


def test_seed_selection_does_not_read_heldout_labels():
    masks = np.ones((3, 2, 40, 40), bool)
    observations = dict(
        accepted=np.ones((3, 2), bool), keypoints_px=np.full((3, 2, 21, 2), 12.0)
    )
    a = select_seeds(observations, masks, [40, 41, 42], [40, 42], [41])
    observations["keypoints_px"][1] = 1e9
    b = select_seeds(observations, masks, [40, 41, 42], [40, 42], [41])
    np.testing.assert_array_equal(a["query_tyx"], b["query_tyx"])
    with pytest.raises(ValueError, match="disjoint"):
        select_seeds(observations, masks, [40, 41, 42], [40, 41], [41])


def test_consensus_rejects_occlusion_drift_and_wrong_hand():
    masks, seeds, tracks, scores = fixture_tracks()
    reverse = tracks.copy()
    scores[0, 1] = 0.2
    reverse[0, 2, 0] += 6
    tracks[:, 3, 0] = 35
    reverse[:, 3, 0] = 35
    result = aggregate_tracks(
        tracks, reverse, scores, np.full_like(scores, 0.95), seeds, masks
    )
    assert result["weights"][0, 0, 4] > 0.9
    assert (result["weights"][1:, 0, 4] == 0).all()
    assert np.isnan(result["keypoints_px"][1:, 0, 4]).all()


def test_disagreement_and_duplicate_seed_times_do_not_count_as_support():
    masks, seeds, tracks, scores = fixture_tracks()
    tracks[1, :, 0] = 24
    result = aggregate_tracks(
        tracks, tracks, scores, scores, seeds, masks, max_agreement_px=5
    )
    assert not result["weights"].any()
    seeds["query_tyx"][:, 0] = 0
    tracks[:] = 10
    result = aggregate_tracks(tracks, tracks, scores, scores, seeds, masks)
    assert not result["weights"].any()


@pytest.mark.parametrize(
    "options", [dict(min_score=float("nan")), dict(max_cycle_px=-1), dict(min_seeds=1)]
)
def test_tracker_rejects_invalid_admission_limits(options):
    masks, seeds, tracks, scores = fixture_tracks()
    with pytest.raises(ValueError):
        aggregate_tracks(tracks, tracks, scores, scores, seeds, masks, **options)


def test_partial_image_evidence_ignores_missing_coordinates_without_nan_loss():
    points = np.full((21, 2), np.nan)
    points[:4] = 12
    clean, weights = image_evidence(points)
    assert np.isfinite(clean).all() and weights.sum() == 4
    points[4:] = 1e30
    clean2, _ = image_evidence(points, weights)
    np.testing.assert_array_equal(clean, clean2)
    with pytest.raises(ValueError, match="four"):
        image_evidence(points, np.zeros(21))
    with pytest.raises(ValueError, match="between"):
        image_evidence(points, np.full(21, -1))


def test_runtime_mounts_exclude_evaluation_and_parent(tmp_path):
    config = dict(
        output=str(tmp_path / "output"),
        parent=str(tmp_path / "parent"),
        reconstruction=str(tmp_path / "reconstruction"),
        frame_start=40,
        frame_count=46,
        heldout_frame_ids=[50, 60],
        mask_ids=[2, 3],
        object_id=1,
        weights=dict(tapir="/weights/tapir", hamer="/weights/hamer"),
    )
    for phase in ["track", "refine"]:
        settings, _ = runtime_config(config, phase)
        paths = settings["read_only_inputs"]
        assert config["parent"] not in paths and config["output"] not in paths
        assert all("evaluation" not in p for p in paths)
    tracking, _ = runtime_config(config, "track")
    assert list(tracking["weights"]) == ["tapir"]
    assert not any("fit_inputs" in p for p in tracking["read_only_inputs"])


def test_confidence_requires_both_visibility_and_low_uncertainty():
    score = confidence(np.array([-20, 20, -20]), np.array([-20, -20, 20]))
    assert score[0] > 0.99 and (score[1:] < 0.01).all()
