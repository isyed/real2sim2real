import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from real2sim2real.reconstruction_repair.appearance_yaw import image_cost, select_path, yaw_pose, landmark_cost, centroid_translation, project_vertex_colors


def test_yaw_keeps_mesh_centroid_and_gravity_heights():
    r = Rotation.from_euler("xyz", [.3, .4, -.6]).as_matrix()
    t, c, up = np.array([.2, -.1, 2.]), np.array([.04, -.02, .01]), np.array([0., -1., 0.])
    candidate, translation = yaw_pose(r, t, c, up, 75.)
    assert np.allclose(candidate@c+translation, r@c+t)
    assert np.allclose(up@candidate, up@r)
    assert np.allclose(candidate@candidate.T, np.eye(3))
    with pytest.raises(ValueError):
        yaw_pose(r, t, c, up*2, 75.)


def test_rgb_feature_disambiguates_matching_round_outline_and_handles_exposure():
    xx, yy = np.meshgrid(np.arange(8, 33), np.arange(8, 33))
    vertices = np.column_stack(((xx.ravel()-20)/20., (yy.ravel()-20)/20., np.zeros(xx.size)))
    colors = np.full((len(vertices), 3), .8)
    dark = (xx.ravel() < 16) & (yy.ravel() >= 15) & (yy.ravel() < 26)
    colors[dark] = .25
    rgb = np.full((41, 41, 3), .8)
    rgb[yy, xx] = colors.reshape(25, 25, 3)*.7+.1
    mask = np.zeros((41, 41), bool)
    mask[yy, xx] = True
    k = np.array([[40., 0, 20], [0, 40, 20], [0, 0, 1.]])
    args = (k, rgb, mask, np.zeros_like(mask))
    good = image_cost(vertices, colors, np.eye(3), [0, 0, 2.], *args)
    wrong = image_cost(vertices, colors, Rotation.from_euler("z", 180, degrees=True).as_matrix(), [0, 0, 2.], *args)
    assert good["usable"] and wrong["usable"]
    assert good["photometric_error"] < 1e-10
    assert wrong["photometric_error"] > .01
    assert not good["candidate_approved"]
    unavailable = image_cost(vertices, colors, np.eye(3), [0, 0, 2.], k, rgb, mask, mask)
    assert not unavailable["usable"] and unavailable["cost"] is None


def test_circular_path_crosses_zero_without_a_fake_270_degree_jump():
    costs = np.full((3, 4), 1.)
    costs[np.arange(3), [0, 3, 2]] = 0.
    path = select_path(costs, [0, 90, 180, 270], [0, 1, 2], max_speed_deg_per_frame=100.)
    assert np.array_equal(path["yaw_deg"], [0., -90., -180.])
    assert path["maximum_fit_step_deg"] == pytest.approx(90.)
    assert not path["candidate_approved"]
    with pytest.raises(ValueError):
        select_path(np.full((3, 4), np.nan), [0, 90, 180, 270], [0, 1, 2])


def test_image_cost_rejects_unlabelled_intensity_units():
    with pytest.raises(ValueError, match="intensity"):
        image_cost(np.array([[0., 0, 1]]), np.full((1, 3), 255.),
                   np.eye(3), [0, 0, 1.], np.eye(3),
                   np.ones((10, 10, 3)), np.ones((10, 10), bool), np.zeros((10, 10), bool))


def test_hidden_landmark_cannot_pass_by_projecting_near_a_visible_pixel():
    points = np.array([[0., 0., -.1]])
    normals = np.array([[0., 0., -1.]])
    k = np.array([[100., 0, 50], [0, 100, 50], [0, 0, 1.]])
    observed = np.array([[50., 50.]])
    good = landmark_cost(points, normals, observed, np.eye(3), [0, 0, 2.], k)
    hidden = landmark_cost(points, normals, observed, Rotation.from_euler("y", 180, degrees=True).as_matrix(), [0, 0, 2.], k)
    assert good["usable"] and good["cost"] == 0
    assert not hidden["usable"] and hidden["cost"] is None
    assert not good["full_self_occlusion_checked"]


def test_centroid_proposal_preserves_height_and_fails_at_metric_bound():
    vertices = np.array([[-.5, -.2, 0], [.5, -.2, 0], [.5, .2, 0], [-.5, .2, 0]])
    mask = np.zeros((50, 50), bool)
    mask[26:35, 11:32] = True
    k = np.array([[40., 0, 20], [0, 40, 20], [0, 0, 1.]])
    args = (vertices, np.eye(3), [0, .5, 2.], k, mask, np.zeros_like(mask), [0., 1., 0.])
    result = centroid_translation(*args)
    assert result["usable"] and result["move_m"] == pytest.approx(.05)
    assert result["height_change_m"] == pytest.approx(0., abs=1e-12)
    assert not result["candidate_approved"]
    assert not centroid_translation(*args, max_move_m=.02)["usable"]


def test_camera_depth_refit_makes_no_gravity_height_claim():
    vertices = np.array([[-.5, -.2, 0], [.5, -.2, 0], [.5, .2, 0], [-.5, .2, 0]])
    mask = np.zeros((50, 50), bool)
    mask[26:35, 11:32] = True
    k = np.array([[40., 0, 20], [0, 40, 20], [0, 0, 1.]])
    result = centroid_translation(vertices, np.eye(3), [0, .5, 2.], k, mask, np.zeros_like(mask),
                                  [0., 0., 1.], plane_semantics="camera_depth")
    assert result["usable"] and result["height_change_m"] is None
    assert result["translation"][2] == 2.
    assert result["plane_coordinate_change_m"] == 0


def test_vertex_preview_does_not_show_a_rear_sample_over_a_front_sample():
    vertices = np.array([[0., 0., 1.], [0., 0., 2.]])
    colors = np.array([[255, 0, 0], [0, 0, 255]], np.uint8)
    k = np.array([[10., 0, 5], [0, 10, 5], [0, 0, 1.]])
    image = project_vertex_colors(vertices, colors, np.eye(3), [0., 0., 0.], k, (10, 10))
    assert np.array_equal(image[5, 5], [255, 0, 0])
