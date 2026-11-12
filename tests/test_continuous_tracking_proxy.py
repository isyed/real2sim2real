import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from real2sim2real.controller_actor_training.continuous_tracking_proxy import (
    interpolate_poses,
    score,
)


def poses(n=3):
    return np.repeat(np.eye(4)[None], n, axis=0)


def test_metric_translation_has_known_keypoint_error_and_threshold_auc():
    a, b = poses(), poses()
    a[:, 0, 3] = 0.025
    s = score(a, b, np.array([[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]]), np.zeros(3))
    assert s["mppe_cm"] == pytest.approx(2.5)
    assert s["add_auc"] == pytest.approx(0.8125)
    assert s["sp_sr"] == 1 and s["mp_sr"] == 1
    a[:, 0, 3] = 0.045
    s = score(a, b, np.zeros((1, 3)), np.zeros(3))
    assert s["sp_sr"] == 1 and s["mp_sr"] == 0


def test_rotation_contributes_to_object_keypoints_without_finger_data():
    a, b = poses(), poses()
    a[:, :3, :3] = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    s = score(a, b, np.eye(3) * 0.05, np.zeros(3))
    assert s["mppe_cm"] == pytest.approx(100 * (4 / 6) * np.sqrt(2) * 0.05)
    assert s["mean_object_position_error_cm"] == 0
    assert s["sp_sr"] == 0 and s["mp_sr"] == 0


def test_recentring_preserves_surface_add_and_scoring_does_not_mutate_inputs():
    a, b = poses(), poses()
    a[:, :3, :3] = Rotation.from_euler("y", 0.2).as_matrix()
    before = a.copy()
    vertices = np.array([[0.05, 0.02, 0], [0.01, 0.05, 0.02], [-0.04, 0, 0.03]])
    origin = score(a, b, vertices, np.zeros(3))
    centered = score(a, b, vertices, vertices.mean(0))
    assert origin["mean_add_cm"] == pytest.approx(centered["mean_add_cm"])
    assert origin["add_auc"] == centered["add_auc"]
    np.testing.assert_array_equal(a, before)


def test_timestamp_sampling_uses_slerp_and_rejects_missing_coverage():
    rotations = Rotation.from_euler("z", [[0], [90]], degrees=True).as_matrix()
    result = interpolate_poses([0, 2], [[0, 0, 0], [2, 0, 0]], rotations, [0, 1, 2])
    np.testing.assert_allclose(result[1, :3, 3], [1, 0, 0])
    np.testing.assert_allclose(
        result[1, :3, :3],
        Rotation.from_euler("z", 45, degrees=True).as_matrix(),
        atol=1e-12,
    )
    with pytest.raises(ValueError, match="coverage"):
        interpolate_poses([0.001, 2], [[0, 0, 0], [2, 0, 0]], rotations, [0, 1])
