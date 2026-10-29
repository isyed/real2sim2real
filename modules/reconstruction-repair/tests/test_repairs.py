import numpy as np
from scipy.spatial.transform import Rotation

from real2sim2real.reconstruction_repair.geometry import (frame_overlap, mesh_audit, mesh_silhouette,
                                               object_centers, points_mesh_distance)
from real2sim2real.reconstruction_repair.repairs import (pose_recovery_propose, rotation_spikes_propose,
                                              scale_propose, semantic_orientation_propose,
                                              temporal_translation_propose, topology_propose)
from real2sim2real.reconstruction_repair.scoring import geometry_scores
from real2sim2real.reconstruction_repair.validation import proposal_validate

from conftest import box_geometry, metadata, observations, only


def validate(b, p, config):
    return proposal_validate(b, b.geometry, p, geometry_scores(b), geometry_scores(b, p.geometry), config)


def test_isolated_orientation_error_repairs_without_moving_center(make_bundle):
    true = box_geometry()
    masks = observations(true)
    broken = {k: v.copy() for k, v in true.items()}
    broken["object_rotation"][4] = Rotation.from_euler("z", 75, degrees=True).as_matrix()
    b = make_bundle(g=broken, o=masks)
    config = only("rotation_spikes")
    p = rotation_spikes_propose(b, b.geometry, config)
    assert p.details["detected_frame_ids"] == [104]
    np.testing.assert_allclose(object_centers(p.geometry), object_centers(b.geometry), atol=1e-12)
    assert frame_overlap(p.geometry, b.observations, 4) > frame_overlap(b.geometry, b.observations, 4) + .5
    assert validate(b, p, config)["accepted"]


def test_real_abrupt_motion_is_not_accepted_as_jitter(make_bundle):
    g = box_geometry()
    g["object_rotation"][4] = Rotation.from_euler("z", 75, degrees=True).as_matrix()
    b = make_bundle(g=g)  # masks support the real movement
    config = only("rotation_spikes")
    p = rotation_spikes_propose(b, g, config)
    v = validate(b, p, config)
    assert not v["accepted"]
    assert "per_frame_silhouette_regressed" in v["reasons"]


def test_pose_repair_is_world_frame_equivariant(make_bundle):
    g = box_geometry()
    o = observations(g)
    g["object_rotation"][4] = Rotation.from_euler("z", 75, degrees=True).as_matrix()
    b = make_bundle(g=g, o=o)
    config = only("rotation_spikes")
    p = rotation_spikes_propose(b, g, config)
    q = Rotation.from_euler("xyz", [30, 15, -25], degrees=True).as_matrix()
    shift = np.array([4., -2., 1.])
    transformed = {k: v.copy() for k, v in g.items()}
    transformed["object_rotation"] = q @ g["object_rotation"]
    transformed["object_translation"] = g["object_translation"] @ q.T + shift
    inverse = np.eye(4)
    inverse[:3, :3], inverse[:3, 3] = q.T, -q.T @ shift
    transformed["world_to_camera"] = g["world_to_camera"] @ inverse
    o2 = {k: v.copy() for k, v in o.items()}
    o2["hand_points_world"] = o["hand_points_world"] @ q.T + shift
    b2 = make_bundle(g=transformed, o=o2)
    p2 = rotation_spikes_propose(b2, transformed, config)
    np.testing.assert_allclose(p2.geometry["object_rotation"], q @ p.geometry["object_rotation"], atol=1e-12)
    np.testing.assert_allclose(p2.geometry["object_translation"], p.geometry["object_translation"] @ q.T + shift, atol=1e-12)
    assert validate(b2, p2, config)["accepted"]


def test_unknown_or_locked_frames_do_not_get_bridged(make_bundle):
    g = box_geometry(11)
    g["object_translation"][:, 0] = [0, .01, -.01, .02, .04, -.06, .02, 0, -.01, .01, 0]
    o = observations(g)
    o["reliable"][5] = False
    o["locked"][8] = True
    b = make_bundle(g=g, o=o)
    p = temporal_translation_propose(b, g, only("temporal_translation"))
    # Each segment keeps its own endpoints. Occlusion and lock themselves stay exact.
    for i in [0, 4, 5, 6, 7, 8, 9, 10]:
        np.testing.assert_array_equal(p.geometry["object_translation"][i], g["object_translation"][i])


def test_time_filter_preserves_linear_motion_with_irregular_timestamps(make_bundle):
    g = box_geometry()
    g["timestamps_s"] = np.array([0, .02, .05, .07, .1, .14, .18, .20, .24])
    g["object_translation"][:, 0] = g["timestamps_s"] * .1
    b = make_bundle(g=g)
    p = temporal_translation_propose(b, g, only("temporal_translation"))
    np.testing.assert_array_equal(p.geometry["object_translation"], g["object_translation"])


def test_temporal_filter_bounded_zero_mean_and_contact_boundary(make_bundle):
    g = box_geometry(15)
    g["object_translation"][1:-1, 0] = .012 * (-1.) ** np.arange(13)
    o = observations(g)
    o["contact_active"][7:] = False
    b = make_bundle(g=g, o=o)
    config = only("temporal_translation")
    p = temporal_translation_propose(b, g, config)
    delta = p.geometry["object_translation"] - g["object_translation"]
    assert np.linalg.norm(delta, axis=1).max() <= config.translation_bound_m + 1e-12
    np.testing.assert_allclose(delta[:7].sum(axis=0), 0, atol=1e-12)
    np.testing.assert_allclose(delta[7:].sum(axis=0), 0, atol=1e-12)
    np.testing.assert_array_equal(delta[[0, 6, 7, 14]], 0)


def test_measured_scale_preserves_center_and_needs_evidence(make_bundle):
    g = box_geometry()
    g["vertices"] += [.1, .04, -.02]  # off-center asset origin
    b = make_bundle(g=g)
    config = only("scale")
    unchanged = scale_propose(b, g, config)
    np.testing.assert_array_equal(unchanged.geometry["object_scale"], g["object_scale"])
    b.manifest["evidence"]["scale"] = {"axis": 0, "length_m": .4, "source": "caliper"}
    p = scale_propose(b, g, config)
    assert abs(p.geometry["object_scale"] - 1.25) < 1e-12
    np.testing.assert_allclose(object_centers(p.geometry), object_centers(g), atol=1e-12)
    # Existing observations support the original size: measured size alone cannot override them.
    assert not validate(b, p, config)["accepted"]


def test_semantic_flip_requires_labels_and_passes_symmetric_outline(make_bundle):
    g = box_geometry()
    o, m = observations(g), metadata()
    o["axis_facing"] = np.ones(9, dtype=np.int8)
    m["evidence"]["semantic_axis_local"] = [0, 0, 1]
    b = make_bundle(g=g, o=o, m=m)
    config = only("semantic_orientation")
    p = semantic_orientation_propose(b, g, config)
    result = validate(b, p, config)
    assert result["accepted"], result
    assert "annotated_orientation_gain" in result["benefits"]
    np.testing.assert_array_equal(p.geometry["object_translation"], g["object_translation"])


def test_contact_regression_rejects_outline_improvement(make_bundle):
    desired = box_geometry()
    original = {k: v.copy() for k, v in desired.items()}
    original["object_translation"][:, 0] = .04
    o = observations(desired)
    o["hand_points_world"][:] = [.20, 0., 2.]
    b = make_bundle(g=original, o=o)
    from real2sim2real.reconstruction_repair.contracts import RepairProposal
    p = RepairProposal("pose_recovery", desired, "Known outline gain, hand separation loss")
    v = validate(b, p, only("pose_recovery"))
    assert not v["accepted"]
    assert "hand_surface_separation_regressed" in v["reasons"]
    gate = next(g for g in v["checked_gates"] if g["id"] == "hand_surface_separation_regressed")
    assert gate["observed"] > gate["threshold"] == 0.005


def test_missing_contact_is_not_treated_as_zero_error(make_bundle):
    g = box_geometry()
    o = observations(g)
    del o["hand_points_world"], o["contact_active"]
    g["object_rotation"][4] = Rotation.from_euler("z", 75, degrees=True).as_matrix()
    b = make_bundle(g=g, o=o)
    cfg = only("rotation_spikes")
    result = validate(b, rotation_spikes_propose(b, g, cfg), cfg)
    assert "contact_evidence_missing" in result["reasons"]
    assert geometry_scores(b).summary["hand_surface_gap_m"]["mean"] is None


def test_point_triangle_distance_not_nearest_vertex():
    vertices = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    values = points_mesh_distance(np.array([[.25, .25, .1], [-.1, 0, 0], [0, 0, 0]]), vertices, np.array([[0, 1, 2]]))
    np.testing.assert_allclose(values, [.1, .1, 0], atol=1e-12)


def test_clean_mesh_no_reindexing_and_packed_ring_hole_preserved(make_bundle):
    b = make_bundle()
    cfg = only("topology")
    clean = topology_propose(b, b.geometry, cfg)
    for key in b.geometry:
        np.testing.assert_array_equal(b.geometry[key], clean.geometry[key])
    n = 16
    theta = np.arange(n) * 2 * np.pi / n
    vertices = np.r_[np.c_[.18*np.cos(theta), .18*np.sin(theta), np.zeros(n)],
                     np.c_[.12*np.cos(theta), .12*np.sin(theta), np.zeros(n)]]
    faces = np.array([[i, (i+1)%n, n+i] for i in range(n)] + [[(i+1)%n, n+(i+1)%n, n+i] for i in range(n)])
    g = box_geometry()
    g["vertices"], g["faces"] = vertices, faces
    expected = mesh_silhouette(g, 0, 96, 128)
    assert not expected[48, 64]
    g["vertices"] = np.r_[vertices, vertices[:1]]
    g["faces"] = np.r_[faces, faces[:1], np.array([[len(vertices), len(vertices), len(vertices)]])]
    b = make_bundle(g=g)
    repaired = topology_propose(b, g, cfg)
    np.testing.assert_array_equal(mesh_silhouette(repaired.geometry, 0, 96, 128), expected)
    assert mesh_audit(repaired.geometry)["boundary_edges"] == 2*n
    assert mesh_audit(repaired.geometry)["euler_characteristic"] == 0
    assert validate(b, repaired, cfg)["accepted"]


def test_pose_recovery_fits_bad_xy_without_camera_depth_change(make_bundle):
    true = box_geometry(9)
    g = {k: v.copy() for k, v in true.items()}
    g["object_translation"][2:7, 0] = .04
    b = make_bundle(g=g, o=observations(true))
    cfg = only("pose_recovery", anchor_spacing_s=.03, max_fit_anchors=5, pose_trigger_iou=.9)
    p = pose_recovery_propose(b, g, cfg)
    assert frame_overlap(p.geometry, b.observations, 4) > frame_overlap(g, b.observations, 4)
    np.testing.assert_array_equal(p.geometry["object_translation"][:, 2], g["object_translation"][:, 2])
    np.testing.assert_array_equal(p.geometry["object_translation"][[0, -1]], g["object_translation"][[0, -1]])
    assert validate(b, p, cfg)["accepted"]
