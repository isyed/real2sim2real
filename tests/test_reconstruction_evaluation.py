import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from real2sim2real.evaluation.geometry import fit_rigid
from real2sim2real.evaluation.trajectory import score


def reference():
    poses = np.repeat(np.eye(4)[None], 5, axis=0)
    poses[:, 0, 3] = np.arange(5) * 0.1
    registration = dict(
        rotation=np.eye(3),
        translation=np.zeros(3),
        scale=1.0,
        reference_center=np.zeros(3),
    )
    return poses, registration, np.eye(3) * 0.1


def test_fixed_initial_alignment_removes_only_world_gauge():
    poses, registration, points = reference()
    world = np.eye(4)
    world[:3, :3] = Rotation.from_euler("z", 30, degrees=True).as_matrix()
    world[:3, 3] = [3, 4, 5]
    metrics, _ = score(world[None] @ poses, poses, registration, points)
    assert metrics["mean_position_cm"] < 1e-10
    assert metrics["mean_orientation_deg"] < 1e-10
    assert metrics["mppe_cm"] < 1e-10
    assert metrics["mp_sr"] and metrics["auc_missing_as_failure"] == pytest.approx(1)


def test_trajectory_scale_error_cannot_be_aligned_away():
    poses, registration, points = reference()
    wrong = poses.copy()
    wrong[:, :3, 3] *= 2
    metrics, _ = score(wrong, poses, registration, points)
    assert metrics["mean_position_cm"] == pytest.approx(20)
    assert not metrics["trajectory_rescaled"] and not metrics["mp_sr"]


def test_missing_predictions_are_explicit_and_penalized():
    poses, registration, points = reference()
    wrong = poses.copy()
    wrong[3:] = np.nan
    metrics, _ = score(wrong, poses, registration, points)
    assert metrics["coverage"] == 0.6
    assert metrics["auc_observed"] == pytest.approx(1)
    assert metrics["auc_missing_as_failure"] == pytest.approx(0.6)
    assert not metrics["sp_sr"] and not metrics["mp_sr"]


def test_initial_pose_is_required():
    poses, registration, points = reference()
    wrong = poses.copy()
    wrong[0] = np.nan
    with pytest.raises(ValueError, match="initial"):
        score(wrong, poses, registration, points)


def test_mesh_body_origin_change_does_not_change_world_geometry():
    poses, registration, points = reference()
    offset = np.array([0.2, -0.3, 0.4])
    wrong = poses.copy()
    wrong[:, :3, 3] -= offset
    registration.update(translation=-offset)
    metrics, _ = score(wrong, poses, registration, points)
    assert metrics["mean_position_cm"] < 1e-10


def test_rigid_fit_has_correct_direction_and_no_reflection():
    points = np.random.default_rng(4).normal(size=(40, 3))
    r = Rotation.from_euler("xyz", [0.4, -0.2, 0.1]).as_matrix()
    t = np.array([1, 2, 3])
    rr, tt = fit_rigid(points, points @ r.T + t)
    assert np.allclose(rr, r) and np.allclose(tt, t)
    assert np.linalg.det(rr) == pytest.approx(1)


def test_auc_integrates_nine_accuracy_thresholds():
    poses, registration, points = reference()
    wrong = poses.copy()
    wrong[1:, 1, 3] = 0.025
    metrics, _ = score(wrong, poses, registration, points)
    # Accuracies at 1 and 2 cm are .2; at 3 through 9 cm they are 1.
    assert metrics["auc_observed"] == pytest.approx(0.85)
