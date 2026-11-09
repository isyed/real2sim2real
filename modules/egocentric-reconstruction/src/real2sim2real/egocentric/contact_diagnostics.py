"""Geometry-only contact candidates, gaps and penetrations; never contact truth."""

import json
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull

from real2sim2real.artifacts import digest, write_json
from .camera import transform
from .interaction_io import Reconstruction, fresh_directory


def support_clearance(vertices, boundary):
    vertices = np.asarray(vertices)
    hull = ConvexHull(np.asarray(boundary))
    inside = (
        vertices[:, :2] @ hull.equations[:, :2].T + hull.equations[:, 2] <= 1e-8
    ).all(1)
    heights = vertices[inside, 2]
    return dict(
        vertices_over_support=int(inside.sum()),
        minimum_height_m=float(heights.min()) if len(heights) else None,
        penetration_depth_m=float(max(0, -heights.min())) if len(heights) else None,
        below_plane_fraction=float((heights < 0).mean()) if len(heights) else None,
    )


def surface_contact(
    mesh,
    points,
    *,
    contact_distance_m=0.01,
    penetration_tolerance_m=0.003,
    proximity=None,
    intersector=None,
):
    """Unsigned distances always; inside/outside only for closed oriented meshes."""
    import trimesh

    _, distances, _ = (
        proximity.closest(points)
        if proximity is not None
        else trimesh.proximity.closest_point(mesh, points)
    )
    result = dict(
        minimum_distance_m=float(distances.min()),
        p10_distance_m=float(np.quantile(distances, 0.1)),
        near_surface_fraction=float((distances <= contact_distance_m).mean()),
        contact_candidate=bool((distances <= contact_distance_m).any()),
        signed_distance_available=bool(
            mesh.is_watertight and mesh.is_winding_consistent
        ),
        penetrating_vertices=None,
        penetration_depth_m=None,
    )
    if result["signed_distance_available"]:
        inside = (
            intersector.contains_points(points)
            if intersector is not None
            else mesh.contains(points)
        )
        penetrates = inside & (distances > penetration_tolerance_m)
        result.update(
            penetrating_vertices=int(penetrates.sum()),
            penetration_depth_m=float(distances[inside].max()) if inside.any() else 0.0,
        )
    return result


def ray_intersector(mesh, backend="auto"):
    if backend == "auto":
        return mesh.ray
    if backend == "triangle":
        from trimesh.ray.ray_triangle import RayMeshIntersector
    elif backend == "embree":
        from trimesh.ray.ray_pyembree import RayMeshIntersector
    else:
        raise ValueError("Ray backend must be auto, triangle or embree")
    return RayMeshIntersector(mesh)


def diagnose(config):
    import trimesh

    source = Reconstruction(config["reconstruction"], config.get("frame_count"))
    root = Path(config["output"])
    support = json.loads((root / "support/support.json").read_text())
    if not support["accepted"]:
        raise ValueError("Cannot diagnose contacts against rejected support")
    scene = np.load(root / "support/scene.npz")
    hand_path = root / "hands_metric/hands.npz"
    hands = np.load(hand_path) if hand_path.exists() else None
    if hands is not None and not np.array_equal(hands["frame_ids"], source.frame_ids):
        raise ValueError("Hand frame IDs differ from the scene")
    out = fresh_directory(
        root / ("contacts" if hands is not None else "contacts-preview")
    )
    object_mesh = trimesh.load(source.path("mesh/metric.glb"), force="mesh")
    from .mesh_surface import MeshSurfaceQuery

    proximity = MeshSurfaceQuery(object_mesh)
    options = config.get("contacts", {})
    intersector = ray_intersector(object_mesh, options.get("ray_backend", "auto"))
    records = []
    for i in range(source.count):
        pose = scene["world_from_object"][i]
        object_vertices = transform(object_mesh.vertices, pose)
        row = dict(
            frame=int(source.frame_ids[i]),
            timestamp_s=float(source.times[i]),
            object_table=support_clearance(
                object_vertices, support["observed_boundary_xy"]
            ),
            hands=[],
        )
        for h, name in enumerate(["left", "right"]):
            record = dict(hand=name, accepted=False, reason="no_accepted_3d_hand")
            if hands is not None and hands["accepted"][i, h]:
                vertices = hands["world_vertices"][i, h]
                local = transform(vertices, np.linalg.inv(pose))
                record = dict(
                    hand=name,
                    accepted=True,
                    table=support_clearance(vertices, support["observed_boundary_xy"]),
                    object=surface_contact(
                        object_mesh,
                        local,
                        proximity=proximity,
                        intersector=intersector,
                        contact_distance_m=options.get("contact_distance_m", 0.01),
                        penetration_tolerance_m=options.get(
                            "penetration_tolerance_m", 0.003
                        ),
                    ),
                )
                tips = hands["world_joints"][i, h, [4, 8, 12, 16, 20]]
                distances = proximity.closest(transform(tips, np.linalg.inv(pose)))[1]
                record["fingertip_distance_m"] = distances.tolist()
            row["hands"].append(record)
        records.append(row)
        if i % 25 == 0:
            print(f"contact diagnostics {i + 1}/{source.count}", flush=True)
    depths = [
        r["object_table"]["penetration_depth_m"]
        for r in records
        if r["object_table"]["penetration_depth_m"] is not None
    ]
    summary = dict(
        schema="real2sim2real.contact-diagnostics/1",
        units="metres",
        total_frames=source.count,
        hand_reconstruction_available=hands is not None,
        accepted_hand_frames=hands["accepted"].sum(0).tolist()
        if hands is not None
        else [0, 0],
        object_table_max_penetration_m=max(depths) if depths else None,
        object_mesh_watertight=bool(object_mesh.is_watertight),
        ray_backend=type(intersector).__module__,
        contact_distance_m=options.get("contact_distance_m", 0.01),
        penetration_tolerance_m=options.get("penetration_tolerance_m", 0.003),
        interpretation="proximity candidates and geometry consistency; no physical contact ground truth",
        object_pose_modified=False,
        hand_pose_modified=False,
        sources=dict(
            support=digest(root / "support/support.json"),
            geometry=digest(source.root / "geometry.npz"),
            hands=digest(hand_path) if hands is not None else None,
        ),
        ground_truth_used=False,
    )
    write_json(out / "diagnostics.json", dict(summary=summary, records=records))
    return summary
