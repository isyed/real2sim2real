import cv2
import numpy as np
import pytest

from real2sim2real.egocentric.multiview_points import (
    epipolar_match,
    reciprocal_match,
    triangulate,
    pixel_sensitivity,
    project_points,
)
from real2sim2real.egocentric.camera import transform
from real2sim2real.egocentric.hand_point_tracks import close_fingertip_pairs


def cameras(baseline=0.075):
    k = np.array([[160.0, 0, 100], [0, 160, 80], [0, 0, 1]])
    t = np.eye(4)
    t[0, 3] = -baseline
    return k, t


def test_coincident_fingertips_warn_without_inventing_identity_or_visibility():
    points = np.full((21, 2), np.nan)
    weights = np.zeros(21)
    points[[4, 8, 12, 16]] = [[40, 50], [70, 20], [71, 20], [75, 22]]
    weights[[4, 8, 16]] = 1
    pairs = close_fingertip_pairs(points, weights)
    assert len(pairs) == 1 and pairs[0]["joints"] == [8, 16]
    assert pairs[0]["requires_identity_review"]
    # An unobserved nearby middle finger does not become measured evidence.
    assert weights[12] == 0 and points[12].tolist() == [71, 20]


def test_triangulation_uses_metric_baseline_and_camera_direction():
    k, t = cameras()
    xyz = np.array([[0.03, -0.01, 0.6]])
    points = [project_points(xyz, k)[0], project_points(transform(xyz, t), k)[0]]
    result = triangulate(points, [(k, np.eye(4)), (k, t)])
    np.testing.assert_allclose(result["xyz_reference_m"], xyz[0], atol=1e-10)
    assert result["positive_depth"] and max(result["reprojection_errors_px"]) < 1e-10
    wrong = triangulate(points, [(k, np.eye(4)), (k, np.linalg.inv(t))])
    assert not wrong["positive_depth"]


def test_small_baseline_increases_pixel_sensitivity():
    result = []
    for baseline in [0.075, 0.02]:
        k, t = cameras(baseline)
        xyz = np.array([[0.03, -0.01, 0.6]])
        points = [project_points(xyz, k)[0], project_points(transform(xyz, t), k)[0]]
        result.append(pixel_sensitivity(points, [(k, np.eye(4)), (k, t)]))
    assert result[1]["max_depth_change_m"] > 3 * result[0]["max_depth_change_m"]


def test_reciprocal_appearance_recovers_disparity_without_model_prior():
    k, t = cameras()
    a = np.random.default_rng(4).integers(0, 256, (160, 200), dtype=np.uint8)
    a = cv2.GaussianBlur(a, (3, 3), 0.5)
    # 20 px disparity corresponds to .6 m with f=160, baseline=.075 m.
    b = np.zeros_like(a)
    b[:, :-20] = a[:, 20:]
    row = reciprocal_match(a, b, [100, 80], k, k, t)
    assert row["accepted"], row
    assert np.linalg.norm(np.array(row["match_px"]) - [80, 80]) < 0.4
    assert row["cycle_error_px"] < 1
    assert not row["anatomical_identity_verified"]


def test_repeated_texture_is_rejected_even_with_high_ncc():
    k, t = cameras()
    x, y = np.meshgrid(np.arange(200), np.arange(160))
    a = ((np.sin(x * np.pi / 4) + np.cos(y * np.pi / 5)) * 50 + 128).astype(np.uint8)
    b = np.roll(a, -20, axis=1)
    row = epipolar_match(a, b, [100, 80], k, k, t)
    assert not row["accepted"]
    assert "ambiguous_photometric_match" in row["reasons"]


def test_textureless_patch_and_out_of_image_are_not_observations():
    k, t = cameras()
    image = np.ones((160, 200), np.uint8) * 100
    row = epipolar_match(image, image, [100, 80], k, k, t)
    assert row["reasons"] == ["low_texture"]
    row = epipolar_match(image, image, [2, 2], k, k, t)
    assert row["reasons"] == ["source_patch_outside_image"]


def test_invalid_calibration_and_pixel_inputs_are_rejected():
    k, t = cameras()
    t[0, 0] = 2
    with pytest.raises(ValueError, match="proper rotation"):
        triangulate([[100, 80], [80, 80]], [(k, np.eye(4)), (k, t)])
    with pytest.raises(ValueError, match="Finite image point"):
        triangulate([[np.nan, 80], [80, 80]], [(k, np.eye(4)), (k, np.eye(4))])
