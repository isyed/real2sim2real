import numpy as np
import pytest
import trimesh
from scipy.spatial.transform import Rotation

from real2sim2real.egocentric.camera import transform
from real2sim2real.egocentric.support import fit_plane, upright_transform, prism_mesh
from real2sim2real.egocentric.contact_diagnostics import (
    support_clearance,
    surface_contact,
    ray_intersector,
)
from real2sim2real.egocentric.metric_hands import (
    fit_hand,
    apply_fit,
    project,
    observed_hand_surface,
)
from real2sim2real.egocentric.runtime import docker_command
from real2sim2real.egocentric.interaction_bundle import validate_hands
from real2sim2real.egocentric.mesh_surface import MeshSurfaceQuery
from real2sim2real.egocentric.hamer_worker import load_fixed_shape
from real2sim2real.egocentric.interaction_summary import summarize_hands


def test_upright_transform_preserves_metric_scene_and_projection():
    rotation = Rotation.from_euler("xy", [35, -12], degrees=True).as_matrix()
    normal = rotation[:, 2]
    origin = np.array([0.1, 0.2, 0.6])
    t = upright_transform(normal, -normal @ origin, origin, origin + normal)
    assert np.allclose(transform([origin], t), [[0, 0, 0]])
    assert np.allclose(t[:3, :3] @ normal, [0, 0, 1])
    camera = np.eye(4)
    camera[:3, 3] = [0.05, 0, -0.2]
    object_pose = np.eye(4)
    object_pose[:3, :3] = rotation
    object_pose[:3, 3] = origin
    assert np.allclose(
        np.linalg.inv(t @ camera) @ (t @ object_pose),
        np.linalg.inv(camera) @ object_pose,
    )
    p = np.array([[0.2, 0.3, 0.8], [-0.2, 0.1, 0.5]])
    assert np.isclose(
        np.linalg.norm(np.diff(transform(p, t), axis=0)),
        np.linalg.norm(np.diff(p, axis=0)),
    )


def test_plane_recovers_level_surface_with_stereo_outliers():
    rng = np.random.default_rng(3)
    points = rng.uniform(-0.3, 0.3, (2000, 3))
    points[:1600, 2] = 0.5 + rng.normal(0, 0.001, 1600)
    normal, offset, inliers = fit_plane(points)
    assert abs(normal[2]) > 0.999
    assert abs(abs(offset) - 0.5) < 0.001
    assert inliers.mean() == pytest.approx(0.8, abs=0.02)
    with pytest.raises(ValueError, match="Too few"):
        fit_plane(points[:10])


def test_collision_slab_is_closed_and_does_not_extend_above_plane():
    mesh = prism_mesh([[-0.3, -0.2], [0.3, -0.2], [0.3, 0.2], [-0.3, 0.2]], 0.04)
    assert mesh.is_watertight and mesh.is_winding_consistent
    assert np.allclose(mesh.bounds[:, 2], [-0.04, 0])
    assert mesh.volume == pytest.approx(0.6 * 0.4 * 0.04)


def test_table_clearance_ignores_points_outside_finite_support():
    boundary = [[-1, -1], [1, -1], [1, 1], [-1, 1]]
    result = support_clearance([[0, 0, -0.02], [0.5, 0.5, 0.1], [2, 2, -10]], boundary)
    assert result["vertices_over_support"] == 2
    assert result["penetration_depth_m"] == pytest.approx(0.02)
    missing = support_clearance([[2, 2, -10]], boundary)
    assert missing["minimum_height_m"] is None


def test_contact_sign_requires_closed_mesh_and_tolerance():
    pytest.importorskip("rtree")
    mesh = trimesh.creation.box(extents=[0.2, 0.2, 0.2])
    report = surface_contact(
        mesh, np.array([[0.105, 0, 0], [0.09, 0, 0], [0.099, 0, 0]])
    )
    assert report["contact_candidate"]
    assert report["penetrating_vertices"] == 1
    assert report["penetration_depth_m"] == pytest.approx(0.01)
    mesh.update_faces(np.arange(len(mesh.faces) - 1))
    report = surface_contact(mesh, np.array([[0.09, 0, 0]]))
    assert not report["signed_distance_available"]
    assert report["penetration_depth_m"] is None


def test_stereo_fit_and_validation_points_are_disjoint_and_invalid_depth_excluded():
    k = np.array([[100.0, 0, 20], [0, 100, 20], [0, 0, 1]])
    depth = np.full((40, 40), 0.5)
    depth[10:15] = 0
    points, holdout = observed_hand_surface(
        depth,
        np.ones_like(depth, bool),
        np.array([[8, 8], [30, 8], [30, 30], [8, 30]]),
        k,
        max_points=10000,
    )
    assert len(points) > 80 and len(holdout) > 80
    assert np.all(points[:, 2] == 0.5)
    assert set(map(tuple, points)).isdisjoint(set(map(tuple, holdout)))


def test_metric_hand_fit_recovers_scale_from_surface_not_virtual_camera():
    pytest.importorskip("rtree")
    rng = np.random.default_rng(4)
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=0.05)
    joints = rng.normal(0, 0.02, (21, 3))
    p = np.array([0.04, -0.06, 0.03, 0.08, -0.04, 0.6, np.log(1.18)])
    vertices = apply_fit(mesh.vertices, p)
    k = np.array([[575.0, 0, 507], [0, 575, 380], [0, 0, 1]])
    image_joints = project(apply_fit(joints, p), k)
    points = vertices[vertices[:, 2] < 0.61]
    estimated = fit_hand(mesh.vertices, mesh.faces, joints, image_joints, points, k)
    assert np.exp(estimated[6]) == pytest.approx(1.18, abs=0.025)
    assert np.linalg.norm(estimated[3:6] - p[3:6]) < 0.004
    assert (
        np.median(
            np.linalg.norm(
                project(apply_fit(joints, estimated), k) - image_joints, axis=1
            )
        )
        < 1
    )


def test_hand_runtime_mounts_only_declared_observations_read_only(tmp_path):
    source, output, inputs, weights = [
        tmp_path / p for p in ("source", "output", "observations", "weights")
    ]
    cfg = dict(
        source_root=str(source),
        weights={"hamer": str(weights)},
        images={"hands": "pinned"},
        read_only_inputs=[str(inputs)],
    )
    command, _ = docker_command(cfg, output, "hands", check=True)
    assert f"{inputs}:{inputs}:ro" in command
    assert "--gpus" not in command
    assert "--network" in command and "none" in command


def test_bundle_rejects_fake_missing_hands_and_wrong_world():
    camera = np.repeat(np.eye(4)[None], 2, axis=0)
    camera[:, :3, 3] = [0.1, -0.2, 0.3]
    scene = dict(
        frame_ids=np.arange(2),
        timestamps_s=np.array([0.0, 0.05]),
        world_from_camera=camera,
    )
    accepted = np.array([[True, False], [False, True]])
    points = np.full((2, 2, 21, 3), np.nan)
    points[accepted] = 0.5
    world = points.copy()
    for i, h in np.argwhere(accepted):
        world[i, h] = transform(points[i, h], camera[i])
    hands = dict(
        frame_ids=scene["frame_ids"],
        timestamps_s=scene["timestamps_s"],
        accepted=accepted,
        camera_joints=points.copy(),
        camera_vertices=points.copy(),
        world_joints=world.copy(),
        world_vertices=world.copy(),
    )
    validate_hands(hands, scene)
    hands["world_joints"][0, 1] = 0
    with pytest.raises(ValueError, match="Missing hands"):
        validate_hands(hands, scene)
    hands["world_joints"] = world.copy()
    hands["world_joints"][0, 0] += 0.01
    with pytest.raises(ValueError, match="share the declared world"):
        validate_hands(hands, scene)
    hands["world_joints"] = world.copy()
    hands["timestamps_s"] = np.array([0.0, 0.051])
    with pytest.raises(ValueError, match="timelines"):
        validate_hands(hands, scene)


@pytest.mark.parametrize("shape", ["sphere", "unequal_triangles"])
def test_accelerated_surface_distances_match_exhaustive_triangle_search(shape):
    rng = np.random.default_rng(42)
    if shape == "sphere":
        mesh = trimesh.creation.icosphere(subdivisions=2, radius=0.1)
    else:
        triangles = rng.normal(size=(80, 3, 3))
        triangles *= np.geomspace(0.0001, 2, 80)[:, None, None]
        triangles += rng.uniform(-1, 1, (80, 1, 3))
        mesh = trimesh.Trimesh(
            triangles.reshape(-1, 3), np.arange(240).reshape(-1, 3), process=False
        )
    points = np.concatenate(
        [rng.normal(0, 0.08, (30, 3)), rng.normal(0, 3, (30, 3)), mesh.vertices[:8]]
    )
    expected = trimesh.proximity.closest_point_naive(mesh, points)
    actual = MeshSurfaceQuery(mesh).closest(points)
    np.testing.assert_allclose(actual[1], expected[1], atol=1e-10, rtol=0)
    np.testing.assert_allclose(
        np.linalg.norm(actual[0] - points, axis=1), actual[1], atol=1e-10, rtol=0
    )


def test_saved_shape_roundtrip_preserves_mano_float32(tmp_path):
    import json

    betas = np.linspace(-0.2, 0.2, 20, dtype=np.float32).reshape(2, 10)
    path = tmp_path / "shape.json"
    path.write_text(json.dumps({"betas": betas.tolist()}))
    loaded = load_fixed_shape(path)
    assert loaded.dtype == np.float32
    np.testing.assert_array_equal(loaded, betas)


@pytest.mark.parametrize("betas", [[[0.0] * 10], [[float("nan")] * 10] * 2])
def test_saved_shape_rejects_invalid_values(tmp_path, betas):
    import json

    path = tmp_path / "shape.json"
    path.write_text(json.dumps({"betas": betas}))
    with pytest.raises(ValueError, match="finite ten-coefficient"):
        load_fixed_shape(path)


def test_summary_keeps_missing_frames_and_exposes_rejected_errors():
    observed = np.array([[True, True], [True, False], [False, True]])
    accepted = np.array([[True, False], [False, False], [False, True]])
    quality, contacts = [], []
    for i in range(3):
        quality.append(
            [
                dict(
                    accepted=bool(accepted[i, h]),
                    reprojection_median_px=(3.0 if accepted[i, h] else 100.0)
                    if observed[i, h]
                    else None,
                    reason="missing_observation" if not observed[i, h] else "image_fit",
                )
                for h in range(2)
            ]
        )
        contacts.append(
            dict(
                hands=[
                    dict(
                        accepted=bool(accepted[i, h]),
                        object=dict(
                            contact_candidate=True,
                            minimum_distance_m=0.005,
                            penetration_depth_m=None,
                        ),
                        table=dict(penetration_depth_m=0.0),
                    )
                    for h in range(2)
                ]
            )
        )
    report = summarize_hands(observed, observed, accepted, quality, contacts)
    for row in report.values():
        assert row["total_frames"] == 3
        assert row["metric_coverage"] == pytest.approx(1 / 3)
        assert row["missing_2d_frames"] == 1
        assert row["rejected_metric_frames"] == 1
        assert (
            row["fit_metrics"]["reprojection_median_px"]["accepted_only"]["maximum"]
            == 3
        )
        assert (
            row["fit_metrics"]["reprojection_median_px"]["all_scored_candidates"][
                "maximum"
            ]
            == 100
        )
        assert row["object_penetration_m"]["maximum"] is None
    contacts[0]["hands"][1]["accepted"] = True
    contacts[2]["hands"][1]["accepted"] = False
    with pytest.raises(ValueError, match="Contact records"):
        summarize_hands(observed, observed, accepted, quality, contacts)


def test_summary_rejects_hands_without_model_observations():
    empty = np.zeros((3, 2), bool)
    with pytest.raises(ValueError, match="upstream observations"):
        summarize_hands(empty, empty, np.ones((3, 2), bool), [], [])


def test_optional_ray_backend_preserves_clear_inside_outside_classification():
    pytest.importorskip("embreex")
    pytest.importorskip("rtree")
    mesh = trimesh.creation.box(extents=[0.2, 0.2, 0.2])
    points = np.array([[0, 0, 0], [0.09, 0.02, 0.03], [0.11, 0, 0], [0.5, 0.5, 0.5]])
    for backend in ("triangle", "embree"):
        np.testing.assert_array_equal(
            ray_intersector(mesh, backend).contains_points(points),
            [True, True, False, False],
        )
    with pytest.raises(ValueError, match="Ray backend"):
        ray_intersector(mesh, "unknown")
