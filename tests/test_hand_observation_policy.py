import numpy as np
import pytest

from real2sim2real.egocentric.hand_observation_policy import (
    observation_reliability,
    policy_options,
    anchor_prior_weights,
    image_normalization_mass,
)
from real2sim2real.egocentric.hand_depth_audit import (
    image_regions,
    residual_summary,
    wrist_halfspace,
)


def example():
    points = np.full((1, 2, 21, 2), np.nan)
    weights = np.zeros((1, 2, 21))
    ids = [2, 3, 4, 8, 16]
    points[0, 0, ids] = [[10, 20], [20, 20], [30, 20], [40, 5], [44, 6]]
    weights[0, 0, ids] = 0.9
    joints = np.ones((1, 2, 21, 3))
    joints[..., :2] = points / 100
    return points, weights, joints, np.diag([100, 100, 1]), np.array([[True, False]])


def test_ambiguity_reduces_both_fingers_without_relabeling_or_filling():
    p, w, j, k, baseline = example()
    saved = p.copy(), w.copy(), j.copy(), baseline.copy()
    r = observation_reliability(p, w, j, k, baseline, options={})
    np.testing.assert_allclose(r["weights"][0, 0, [8, 16]], 0.09)
    np.testing.assert_allclose(r["weights"][0, 0, [2, 3, 4]], 0.9)
    assert r["ambiguous"][0, 0, 5:9].all() and r["ambiguous"][0, 0, 13:17].all()
    assert np.array_equal(r["weights"] > 0, w > 0)
    for actual, before in zip((p, w, j, baseline), saved):
        np.testing.assert_array_equal(actual, before)


def test_anchor_reliability_uses_unambiguous_fit_residuals_only():
    p, w, j, k, baseline = example()
    j[0, 0, [8, 16], 0] += 1  # Disputed tips cannot spoil anchor assessment.
    r = observation_reliability(p, w, j, k, baseline, options={})
    assert r["reliable_baseline"].tolist() == [[True, False]]
    j[0, 0, [2, 3, 4], 0] += 0.2  # Genuine 20 px conflict in retained evidence.
    r = observation_reliability(p, w, j, k, baseline, options={})
    assert not r["reliable_baseline"].any()
    assert r["records"][0][0]["initial_unambiguous_median_px"] == pytest.approx(20)


def test_insufficient_unambiguous_evidence_does_not_make_anchor_reliable():
    p, w, j, k, baseline = example()
    w[0, 0, 2] = 0
    r = observation_reliability(p, w, j, k, baseline, options={})
    assert not r["reliable_baseline"].any()


def test_weak_anchor_prior_preserves_recovered_pose_prior_strength():
    np.testing.assert_array_equal(
        anchor_prior_weights([True, True, False], [True, False, False], 0.25),
        [1, 0.25, 1],
    )


def test_downweighting_does_not_amplify_remaining_image_terms():
    original = np.ones((2, 21))
    fitted = original.copy()
    fitted[:, 5:] *= 0.1
    np.testing.assert_array_equal(image_normalization_mass(fitted, original), [21, 21])
    with pytest.raises(ValueError):
        image_normalization_mass(original, fitted)


@pytest.mark.parametrize(
    "options",
    [
        dict(ambiguous_weight=0),
        dict(ambiguous_weight=2),
        dict(reassess_anchors=1),
        dict(anchor_min_landmarks=2),
        dict(weak_anchor_prior_weight=float("nan")),
    ],
)
def test_invalid_policy_is_rejected(options):
    with pytest.raises(ValueError):
        policy_options(options)


def test_depth_regions_are_fixed_image_neighborhoods_with_explicit_complement():
    xyz = np.array([[0, 0, 1], [0.1, 0, 1], [0.5, 0, 1]])
    uv = np.array([[0, 0], [np.nan, np.nan]])
    regions = image_regions(
        xyz, np.diag([100, 100, 1]), uv, {"visible": [0], "missing": [1]}, radius_px=12
    )
    assert regions["visible"].tolist() == [True, True, False]
    assert not regions["missing"].any()
    assert regions["remaining"].tolist() == [False, False, True]
    before = residual_summary(np.array([0.001, 0.002, 0.04]), regions)
    after = residual_summary(np.array([0.002, 0.003, 0.02]), regions)
    assert before["visible"]["count"] == after["visible"]["count"] == 2
    assert (
        before["remaining"]["over_30mm"] == 1 and after["remaining"]["over_30mm"] == 0
    )
    assert before["missing"]["median_m"] is None


def test_wrist_boundary_uses_observed_palm_direction_and_keeps_margin():
    landmarks = np.zeros((21, 2))
    landmarks[[5, 9, 13, 17]] = [0, -50]
    points = np.array([[0, -0.2, 1], [0, 0.05, 1], [0, 0.2, 1]])
    regions = wrist_halfspace(points, np.diag([100, 100, 1]), landmarks, margin_px=10)
    assert regions["beyond_detected_wrist"].tolist() == [False, False, True]
    np.testing.assert_array_equal(
        regions["within_detected_hand_side"], ~regions["beyond_detected_wrist"]
    )
    landmarks[0] = np.nan
    with pytest.raises(ValueError, match="Observed wrist"):
        wrist_halfspace(points, np.eye(3), landmarks)
