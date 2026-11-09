import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from real2sim2real.egocentric.camera import (
    rigid,
    transform,
    unproject,
    project_depth,
    resize_intrinsic,
)
from real2sim2real.egocentric.stereo import Stereo, refine_calibration


def test_depth_project_unproject_and_nearest_surface():
    k = np.array([[100.0, 0, 2], [0, 100, 1], [0, 0, 1]])
    depth = np.full((3, 5), 2.0)
    xyz = unproject(depth, k)
    assert np.allclose(project_depth(xyz, k, (5, 3)), depth)
    assert np.allclose(
        project_depth(
            np.concatenate([xyz.reshape(-1, 3), xyz.reshape(-1, 3) * 2]), k, (5, 3)
        ),
        depth,
    )


def test_invalid_depth_does_not_become_geometry():
    points = np.array([[0, 0, -1], [np.nan, 0, 1], [0, 0, 2]])
    assert np.array_equal(project_depth(points, np.eye(3), (2, 2)), [[2, 0], [0, 0]])


def test_resize_preserves_pixel_centres():
    k = np.array([[100.0, 0, 50], [0, 100, 30], [0, 0, 1]])
    result = resize_intrinsic(k, (100, 60), (50, 30))
    assert np.allclose(result, [[50, 0, 24.75], [0, 50, 14.75], [0, 0, 1]])


def test_moving_camera_composition_cancels_camera_motion():
    world_from_camera = np.eye(4)
    world_from_camera[:3, :3] = Rotation.from_euler("y", 35, degrees=True).as_matrix()
    world_from_camera[:3, 3] = [0.4, 0.1, 0.2]
    world_from_object = np.eye(4)
    world_from_object[:3, 3] = [0.3, 0.2, 1]
    camera_from_object = np.linalg.inv(world_from_camera) @ world_from_object
    assert np.allclose(world_from_camera @ camera_from_object, world_from_object)
    assert np.allclose(
        transform([[0, 0, 0]], world_from_camera @ camera_from_object), [[0.3, 0.2, 1]]
    )


@pytest.mark.parametrize(
    "matrix", [np.diag([-1, 1, 1, 1]), np.diag([2, 2, 2, 1]), np.full((4, 4), np.nan)]
)
def test_camera_pose_rejects_reflections_scale_and_nan(matrix):
    with pytest.raises(ValueError):
        rigid(matrix)


def test_stereo_order_must_match_disparity():
    left = dict(
        size=[64, 48],
        intrinsics=[[50, 0, 32], [0, 50, 24], [0, 0, 1]],
        distortion=[0] * 5,
        rig_from_camera=np.eye(4).tolist(),
    )
    right = dict(left)
    pose = np.eye(4)
    pose[0, 3] = 0.075
    right["rig_from_camera"] = pose.tolist()
    model = Stereo(left, right, left, disparities=16)
    assert model.projections[1][0, 3] < 0
    with pytest.raises(ValueError, match="ordering"):
        Stereo(right, left, left, disparities=16)


def test_calibration_cannot_validate_on_fit_frames():
    with pytest.raises(ValueError, match="Disjoint"):
        refine_calibration({}, [0, 1], [1, 2])
