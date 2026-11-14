import numpy as np
import cv2
import pytest
from real2sim2real.egocentric.multiview_tracks import (
    stereo_to_target,
    common_offset_errors,
    three_camera_identities,
)


def test_triangulation_and_distorted_third_view_without_rgb_fit():
    k = np.array([[500.0, 0, 320], [0, 510, 240], [0, 0, 1]])

    def camera(x):
        t = np.eye(4)
        t[0, 3] = x
        return dict(
            intrinsics=k,
            distortion=np.array([0.03, -0.01, 0.001, -0.002, 0.0]),
            rig_from_camera=t,
        )

    cams = [camera(x) for x in [0, 0.075, -0.03]]
    p = np.random.default_rng(4).uniform([-0.2, -0.1, 0.5], [0.2, 0.1, 2], (40, 3))
    uv = [
        cv2.projectPoints(
            p - s["rig_from_camera"][:3, 3],
            np.zeros(3),
            np.zeros(3),
            k,
            s["distortion"],
        )[0][:, 0]
        for s in cams
    ]
    xyz, target, valid = stereo_to_target(*cams, uv[0], uv[1])
    assert (
        valid.all()
        and np.allclose(xyz, p, atol=1e-7)
        and np.allclose(target, uv[2], atol=1e-6)
    )
    cams[1]["rig_from_camera"][0, 3] *= -1
    _, target, valid = stereo_to_target(*cams, uv[0], uv[1])
    assert not valid.any() and np.isnan(target).all()


def test_offset_comparison_uses_common_identities_and_known_shift():
    p = np.zeros((7, 3, 2))
    p[:, :, 0] = np.arange(7)[:, None] * 2
    q = p.copy()
    q[:, :, 0] -= 2
    valid = np.ones((7, 3), bool)
    valid[0, 1] = False
    valid[6, 2] = False
    common, error = common_offset_errors(p, q, [1, 2, 3, 4, 5], [-1, 0, 1], valid)
    assert common.tolist() == [True, False, False]
    assert np.all(error[2] == 0) and np.all(error[1, :, :, 0] == -2)
    with pytest.raises(ValueError):
        common_offset_errors(p, q, [0], [-1, 0], valid)


def test_cycle_identity_deduplicates_orientations_and_keeps_empty_shape():
    p = np.array([[10.0, 10], [10.0, 10], [20.0, 20]])
    d = np.array([[0.0, 0], [10, 10], [20, 20]], np.float32)
    rows = three_camera_identities([(p, d), (p + 1, d), (p + 2, d)])
    assert rows.shape == (2, 3)
    assert rows.tolist() == [[0, 0, 0], [2, 2, 2]]
