import numpy as np
import pytest

from real2sim2real.egocentric.hand_temporal import (
    component_admission,
    neighbor_target,
    temporal_measurements,
    visibility_weights,
)
from real2sim2real.egocentric.hand_observation_probe import restore_rotation, consensus
from real2sim2real.egocentric.mano_refinement import damped_parameters
from real2sim2real.egocentric.interaction_readiness import coverage_assessment


def test_2d_candidates_never_count_as_metric_recovery_or_control_readiness():
    detected = np.ones((5, 2), bool)
    detected[2, 1] = False
    accepted = detected.copy()
    accepted[3, 0] = False
    candidates = np.zeros_like(accepted)
    candidates[2, 1] = True
    result = coverage_assessment(
        accepted, detected, candidates, np.arange(5), np.arange(5) * 0.05
    )
    assert result["hands"]["right"]["accepted_metric_frames"] == 4
    assert result["hands"]["right"]["missing_2d_after_probe"] == 0
    assert (
        result["hands"]["left"][
            "remaining_metric_gaps_even_if_all_new_2d_candidates_pass_3d"
        ]
        == 1
    )
    assert not result["control_ready"]
    assert result["decision"] == "not_ready_for_full_sequence_control"
    complete = coverage_assessment(
        np.ones_like(accepted),
        np.ones_like(detected),
        np.zeros_like(candidates),
        np.arange(5),
        np.arange(5) * 0.05,
    )
    assert not complete["control_ready"]
    assert complete["decision"] == "requires_robot_specific_validation"


def test_backtracking_scales_increment_not_the_parent_pose_or_metric_scale():
    fit = np.array([0, 0, 0.3, 0.1, 0.2, 0.5, np.log(0.8)])
    initial = np.tile([0, 0, 0.2], (15, 1))
    details = dict(
        incremental_local_rotation_delta_rad=np.tile([0, 0, 0.1], (15, 1)),
        rigid_rotation_delta_rad=[0, 0, 0.2],
        translation_delta_m=[0.002, 0, 0],
    )
    updated, local = damped_parameters(fit, initial, details, 0.5)
    np.testing.assert_allclose(updated, [0, 0, 0.4, 0.101, 0.2, 0.5, np.log(0.8)])
    np.testing.assert_allclose(local, np.tile([0, 0, 0.25], (15, 1)))
    np.testing.assert_allclose(initial, np.tile([0, 0, 0.2], (15, 1)))


@pytest.mark.parametrize("turns", range(4))
def test_rotated_image_coordinates_return_to_original_pixels(turns):
    image = np.arange(35).reshape(5, 7)
    rotated = np.rot90(image, turns)
    y, x = np.indices(rotated.shape)
    restored = restore_rotation(np.c_[x.ravel(), y.ravel()], 7, 5, turns).astype(int)
    np.testing.assert_array_equal(
        image[restored[:, 1], restored[:, 0]], rotated.ravel()
    )


def test_detection_consensus_requires_different_views_and_matching_articulation():
    a = dict(keypoints_px=np.zeros((21, 2)), view=[0, 0], mask_overlap=1)
    assert consensus([a, a]) is None
    assert (
        consensus([a, dict(a, view=[0, 1], keypoints_px=np.ones((21, 2)))])[
            "agreeing_views"
        ]
        == 2
    )
    assert (
        consensus([a, dict(a, view=[0, 1], keypoints_px=np.ones((21, 2)) * 50)]) is None
    )


def test_visibility_never_reads_heldout_depth_and_labels_unsupported_pixels():
    depth = np.full((30, 30), 0.5)
    mask = np.ones_like(depth, bool)
    obj = np.zeros_like(mask)
    uv = np.tile([15.0, 15.0], (21, 1))
    joints = np.tile([0, 0, 0.52], (21, 1))
    before = visibility_weights(uv, joints, depth, mask, obj)
    yy, xx = np.indices(depth.shape)
    depth[(xx // 3 + yy // 3) % 2 == 1] = 1.49
    after = visibility_weights(uv, joints, depth, mask, obj)
    np.testing.assert_array_equal(before[0], after[0])
    assert before[1] == after[1] == ["mask_and_fit_depth_supported"] * 21
    mask[:] = False
    obj[:] = True
    weights, labels = visibility_weights(uv, joints, depth, mask, obj)
    assert np.all(weights == 0.2) and labels == ["object_occluded"] * 21
    uv[0] = [-20, 15]
    assert visibility_weights(uv, joints, depth, mask, obj)[1][0] == "outside_image"


def test_nonuniform_timestamps_give_zero_acceleration_for_constant_velocity():
    times = np.array([0, 0.03, 0.08, 0.14])
    joints = np.zeros((4, 2, 21, 3))
    joints[..., 0] = times[:, None, None] * 0.2
    accepted = np.ones((4, 2), bool)
    velocity, acceleration, edges, triples = temporal_measurements(
        joints, accepted, times
    )
    np.testing.assert_allclose(velocity[..., 0], 0.2)
    np.testing.assert_allclose(acceleration, 0, atol=1e-12)
    assert edges.all() and triples.all()
    np.testing.assert_allclose(
        neighbor_target(joints, accepted, times, 1, 0), joints[1, 0]
    )


def test_temporal_prior_never_bridges_missing_hands_or_large_capture_gaps():
    accepted = np.ones((5, 2), bool)
    accepted[2, 0] = False
    joints = np.zeros((5, 2, 21, 3))
    joints[~accepted] = np.nan
    times = np.arange(5) * 0.05
    assert neighbor_target(joints, accepted, times, 1, 0) is None
    assert neighbor_target(joints, accepted, times, 2, 0) is None
    _, _, _, triples = temporal_measurements(joints, accepted, times)
    assert not triples[:, 0].any() and triples[:, 1].all()
    assert neighbor_target(joints, accepted, times * 10, 1, 1) is None
    assert neighbor_target(joints, accepted, times, 0, 1) is None


def test_temporal_transaction_accepts_improvement_and_rejects_regression():
    accepted = np.ones((7, 2), bool)
    before = np.zeros((7, 2, 21, 3))
    before[3, :, :, 0] = 0.03
    proposed = before.copy()
    proposed[3, 0] = 0
    proposed[3, 1, :, 0] = 0.06
    selected = np.zeros_like(accepted)
    selected[3] = True
    mask, report = component_admission(
        before, proposed, accepted, selected, np.arange(7) * 0.05
    )
    assert mask[3].tolist() == [True, False]
    assert mask.sum() == 1
    assert report[0]["acceleration_rms_after_m_s2"] == 0
    assert "peak_acceleration_regression" in report[1]["reasons"]
    accepted[3, 0] = False
    with pytest.raises(ValueError, match="accepted parent"):
        component_admission(before, proposed, accepted, selected, np.arange(7) * 0.05)
