import numpy as np
import pytest
import trimesh

from real2sim2real.retargeting.contact_site_search import (
    sample_sites,
    shortlist_sites,
    upward_force_capacity,
)


def test_surface_sampling_is_repeatable_and_covers_opposite_faces():
    mesh = trimesh.creation.box(extents=[0.12, 0.08, 0.003])
    sites = sample_sites(mesh, count=12)
    assert sites == sample_sites(mesh, count=12)
    assert len({s["face_index"] for s in sites}) == 12
    assert {tuple(s["normal_object"]) for s in sites} == {
        tuple(n) for n in mesh.face_normals
    }
    rotated = mesh.copy()
    rotation = trimesh.transformations.rotation_matrix(0.7, [0.2, 0.3, 0.5])
    rotated.apply_transform(rotation)
    # Distances combine metric positions and normals, without a preferred world up.
    before = np.array([s["point_object_m"] for s in sites])
    after = np.array([s["point_object_m"] for s in sample_sites(rotated, count=12)])
    assert len(after) == len(before)
    with pytest.raises(ValueError):
        sample_sites(mesh, count=13)
    with pytest.raises(ValueError):
        sample_sites(mesh, normal_distance_m=float("nan"))


def test_shortlist_uses_discovery_only_and_retains_opposed_sheet_faces():
    sites = [
        dict(site_id=i, point_object_m=p, normal_object=n)
        for i, (p, n) in enumerate(
            [
                ([0, 0, 0], [0, 0, 1]),
                ([0.001, 0, 0], [0, 0, 1]),
                ([0, 0, 0.003], [0, 0, -1]),
                ([0.1, 0, 0], [0, 0, 1]),
            ]
        )
    ]
    summaries = [
        dict(
            split="discovery",
            evidence="supported",
            observed_frames=5 - i,
            p90_abs_depth_error_m=0.004,
        )
        for i in range(4)
    ]
    summaries[3]["evidence"] = "insufficient"
    assert shortlist_sites(sites, summaries) == [0, 2]
    summaries[0]["split"] = "validation"
    with pytest.raises(ValueError, match="discovery-only"):
        shortlist_sites(sites, summaries)


def test_no_supported_discovery_does_not_create_a_fallback_shortlist():
    sites = [dict(site_id=0, point_object_m=[0, 0, 0], normal_object=[0, 0, 1])]
    assert (
        shortlist_sites(sites, [dict(split="discovery", evidence="insufficient")]) == []
    )


def test_lifting_screen_distinguishes_visible_top_from_supporting_underside():
    np.testing.assert_allclose(
        upward_force_capacity([[0, 0, 1], [0, 0, -1], [1, 0, 0]], 0.8), [-1, 1, 0.8]
    )
    mesh = trimesh.creation.box()
    eligible = mesh.face_normals[:, 2] < 0
    sites = sample_sites(mesh, count=2, eligible_faces=eligible)
    assert all(s["normal_object"][2] == -1 for s in sites)
    with pytest.raises(ValueError):
        sample_sites(mesh, count=2, eligible_faces=eligible.astype(int))
    with pytest.raises(ValueError):
        upward_force_capacity([[0, 0, 2]], 0.8)
    with pytest.raises(ValueError):
        upward_force_capacity([[0, 0, 1]], -0.1)


def test_arrangement_excludes_failed_validation_and_deduplicates_sites(tmp_path):
    import json
    from real2sim2real.artifacts import digest
    from real2sim2real.controller_actor_training.support_arrangement import combine

    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            dict(
                candidates=[
                    dict(
                        pair=dict(center_of_mass_m=[0, 0, 0]),
                        pad_plan=dict(
                            point_wrench_baseline=dict(
                                mass_kg=0.2, friction=0.8, max_normal_force_n=10
                            )
                        ),
                    )
                ]
            )
        )
    )
    config = dict(
        plan=str(plan),
        plan_sha256=digest(plan),
        capture_sha256="capture",
        calibration_sha256="calibration",
        observation_limits={},
    )
    sites = [
        dict(
            site_id=i,
            face_index=i,
            shortlisted=True,
            discovery=dict(evidence="supported"),
            validation=dict(evidence="supported" if i < 2 else "inconsistent"),
            normal_task_world=[0, 0, -1],
            point_task_world_m=[(-1 if i == 0 else 1) * 0.02, 0, 0],
        )
        for i in range(3)
    ]
    path = tmp_path / "survey.json"
    path.write_text(json.dumps(dict(config=config, sites=sites)))
    args = dict(
        surveys=[dict(path=str(path), sha256=digest(path))] * 2,
        output=str(tmp_path / "out"),
    )
    result = combine(args)
    assert len(result["sites"]) == 2
    assert result["optimistic_upward_force_bound_n"] == 20
    assert result["gravity_wrench"]["feasible"]
    assert not result["control_ready"] and not result["finite_patch_validated"]
    args["surveys"][0]["sha256"] = "changed"
    with pytest.raises(ValueError, match="changed"):
        combine(args)


def test_inner_wall_sampling_excludes_outer_wall_and_top_and_is_world_consistent():
    from real2sim2real.retargeting.contact_site_search import inner_wall_faces

    ring = trimesh.creation.annulus(r_min=0.02, r_max=0.04, height=0.01, sections=24)
    pose = np.eye(4)
    selected = inner_wall_faces(ring, pose, [0, 0])
    assert selected.any()
    assert np.max(np.linalg.norm(ring.triangles_center[selected, :2], axis=1)) < 0.021
    assert np.max(abs(ring.face_normals[selected, 2])) < 1e-6
    pose[:3, 3] = [0.2, -0.1, 0.3]
    np.testing.assert_array_equal(inner_wall_faces(ring, pose, [0.2, -0.1]), selected)
    pose[0, 0] = 2
    with pytest.raises(ValueError, match="Rigid"):
        inner_wall_faces(ring, pose, [0.2, -0.1])
    with pytest.raises(ValueError):
        inner_wall_faces(ring, np.eye(4), [float("nan"), 0])
