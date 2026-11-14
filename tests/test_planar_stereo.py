import numpy as np
import pytest
from real2sim2real.egocentric.planar_stereo import plane_points, sample_pair


def test_intersection_and_stereo_warp_on_sloped_plane():
    k = np.array([[500.0, 0, 40], [0, 500, 30], [0, 0, 1]])
    T = np.eye(4)
    T[:3, :3] = np.diag([1.0, -1.0, -1.0])
    T[:3, 3] = [0, 0, 0.7]
    uv = np.array([[40.0, 30.0], [55, 40], [25, 20]])
    xyz, depth = plane_points(uv, k, T, [0.1, -0.1, 0.02])
    np.testing.assert_allclose(
        xyz[:, 2], 0.1 * xyz[:, 0] - 0.1 * xyz[:, 1] + 0.02, atol=1e-12
    )
    np.testing.assert_allclose(
        (xyz - T[:3, 3]) @ T[:3, :3],
        np.c_[(uv - [40, 30]) / 500, np.ones(3)] * depth[:, None],
    )
    xx = np.indices((100, 100))[1].astype(float)
    view = {
        "left": xx,
        "right": xx,
        "pixels": uv,
        "intrinsics": k,
        "world_from_camera": T,
        "focal_baseline_m_px": 10.0,
    }
    a, b, d = sample_pair(view, [0.1, -0.1, 0.02])
    np.testing.assert_allclose(a - b, d)


def test_stereo_border_is_not_accepted_as_matching_evidence():
    view = {
        "left": np.zeros((20, 20)),
        "right": np.zeros((20, 20)),
        "pixels": np.array([[1.0, 10.0]]),
        "intrinsics": np.eye(3),
        "world_from_camera": np.eye(4),
        "focal_baseline_m_px": 10.0,
    }
    with pytest.raises(ValueError, match="outside image"):
        sample_pair(view, [0.0, 0.0, 1.0])
