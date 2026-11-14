import numpy as np
import pytest

from real2sim2real.egocentric.observed_scene import (
    finger_probe,
    polygon_mask,
    reciprocal_disparity_mask,
    sample_visibility,
    surface_mesh,
    voxel_points,
)


def test_disparity_requires_reverse_match_and_inside_image():
    left = np.full((2, 10), 3.0)
    right = left.copy()
    right[:, 3] = 6
    passed, _ = reciprocal_disparity_mask(left, right)
    assert not passed[:, :3].any()  # projection outside right image
    assert not passed[:, 6].any()  # reverse disparity inconsistent
    assert passed[:, 7:].all()
    left[0, 8] = np.nan
    assert not reciprocal_disparity_mask(left, right)[0][0, 8]


def test_surface_mesh_never_bridges_unknown_pixel_or_depth_discontinuity():
    y, x = np.indices((4, 6))
    points = np.stack([x * 0.002, y * 0.002, np.ones_like(x)], -1)
    points[:, 3:, 2] += 0.1
    valid = np.ones((4, 6), bool)
    valid[1, 1] = False
    mesh = surface_mesh(points, valid, maximum_edge_m=0.01)
    assert len(mesh.faces)
    assert (
        np.linalg.norm(
            mesh.triangles - np.roll(mesh.triangles, 1, axis=1), axis=2
        ).max()
        < 0.01
    )
    assert not np.isclose(mesh.vertices, points[1, 1]).all(1).any()
    before = valid.copy()
    points[0, 0] = np.nan
    surface_mesh(points, valid)
    assert np.array_equal(valid, before)  # input evidence is immutable


def test_no_geometry_when_all_observations_are_missing():
    mesh = surface_mesh(np.full((3, 3, 3), np.nan), np.zeros((3, 3), bool))
    assert len(mesh.vertices) == len(mesh.faces) == 0


def test_unknown_depth_is_never_free_and_occluded_is_not_free():
    projection = np.c_[np.eye(3), np.zeros(3)]
    depth = np.ones((4, 4))
    valid = np.ones_like(depth, bool)
    valid[1, 1] = False
    samples = np.array(
        [[0, 0, 0.5], [0, 0, 1.1], [0.5, 0.5, 0.5], [10, 0, 0.5], [0, 0, -1]]
    )
    result = sample_visibility(samples, np.eye(4), projection, depth, valid)
    assert result.tolist() == [1, 2, 0, 0, 0]
    pose = np.eye(4)
    pose[:3, 3] = [0.2, 0.3, 0.4]
    assert np.array_equal(
        sample_visibility(samples + pose[:3, 3], pose, projection, depth, valid), result
    )


def test_probe_has_metric_bound_and_excludes_contact_endpoint():
    center = np.array([0.1, 0.2, 0.3])
    normal = np.array([0.0, 1.0, 0.0])
    p = finger_probe(center, normal)
    longitudinal = (p - center) @ normal
    radial = p - center - longitudinal[:, None] * normal
    assert np.allclose([longitudinal.min(), longitudinal.max()], [0.006, 0.03])
    assert np.linalg.norm(radial, axis=1).max() <= 0.005 + 1e-12
    with pytest.raises(ValueError, match="unit normal"):
        finger_probe(center, [0, 2, 0])


def test_voxel_thinning_keeps_measured_points_not_average_in_empty_space():
    points = np.array([[0.0001, 0, 1], [0.0019, 0, 1], [0.0041, 0, 1]])
    assert np.array_equal(voxel_points(points), points[[0, 2]])


def test_image_regions_cannot_wrap_around_frame():
    with pytest.raises(ValueError, match="outside"):
        polygon_mask((10, 10), [[[-1, 0], [3, 0], [3, 3]]])
    m = polygon_mask((10, 10), [[[1, 1], [8, 1], [8, 8], [1, 8]]], erosion_pixels=1)
    assert m.sum() == 36
