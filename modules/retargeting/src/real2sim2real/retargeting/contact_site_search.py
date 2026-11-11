"""Geometry coverage and evidence-first selection of provisional support sites."""

import numpy as np


def sample_sites(mesh, *, count=192, normal_distance_m=0.01, eligible_faces=None):
    """Deterministic farthest-point coverage of face centers and normal directions.

    Normal distance keeps opposite faces of a thin sheet distinct. These are
    hypotheses, not measured contacts or verified finite-area patches.
    """
    if (
        type(count) is not int
        or not 2 <= count <= 512
        or not np.isfinite(normal_distance_m)
        or normal_distance_m <= 0
    ):
        raise ValueError("Bounded site count and positive normal distance required")
    eligible = (
        np.ones(len(mesh.faces), bool)
        if eligible_faces is None
        else np.asarray(eligible_faces)
    )
    if eligible.shape != (len(mesh.faces),) or eligible.dtype.kind != "b":
        raise ValueError("Boolean face eligibility required")
    valid = np.flatnonzero((mesh.area_faces > 1e-12) & eligible)
    p = np.asarray(mesh.triangles_center)[valid]
    n = np.asarray(mesh.face_normals)[valid]
    if len(valid) < count or not np.isfinite(p).all() or not np.isfinite(n).all():
        raise ValueError("Enough finite nondegenerate source faces required")
    features = np.c_[p, n * normal_distance_m]
    selected = []
    minimum = np.full(len(p), np.inf)
    index = int(np.argmax(mesh.area_faces[valid]))
    for _ in range(count):
        selected.append(index)
        minimum = np.minimum(minimum, np.sum((features - features[index]) ** 2, axis=1))
        minimum[selected] = -1
        index = int(np.argmax(minimum))
    return [
        dict(
            site_id=i,
            face_index=int(valid[j]),
            point_object_m=p[j].tolist(),
            normal_object=n[j].tolist(),
        )
        for i, j in enumerate(selected)
    ]


def upward_force_capacity(outward_normals, friction):
    """Maximum upward force per unit compressive normal force, circular cone.

    Positive capacity is necessary, not sufficient, for supporting gravity.
    This screen cannot establish torque balance, reachability or contact area.
    """
    normals = np.asarray(outward_normals, float)
    if (
        normals.ndim != 2
        or normals.shape[1] != 3
        or not np.isfinite(normals).all()
        or not np.allclose(np.linalg.norm(normals, axis=1), 1)
        or not np.isfinite(friction)
        or friction < 0
    ):
        raise ValueError(
            "Unit surface normals and finite nonnegative friction required"
        )
    return -normals[:, 2] + friction * np.sqrt(np.maximum(0, 1 - normals[:, 2] ** 2))


def shortlist_sites(sites, summaries, *, maximum=16, minimum_spacing_m=0.008):
    """Rank using discovery evidence only; never accept a held-out record here."""
    if (
        not 1 <= maximum <= 32
        or minimum_spacing_m <= 0
        or not np.isfinite(minimum_spacing_m)
    ):
        raise ValueError("Bounded shortlist and positive metric spacing required")
    if any(s.get("split") != "discovery" for s in summaries) or len(summaries) != len(
        sites
    ):
        raise ValueError("One discovery-only summary per site required")
    eligible = [i for i, s in enumerate(summaries) if s["evidence"] == "supported"]
    eligible.sort(
        key=lambda i: (
            -summaries[i]["observed_frames"],
            summaries[i]["p90_abs_depth_error_m"],
            sites[i]["site_id"],
        )
    )
    selected = []
    for i in eligible:
        p = np.asarray(sites[i]["point_object_m"])
        n = np.asarray(sites[i]["normal_object"])
        if any(
            np.linalg.norm(p - np.asarray(sites[j]["point_object_m"]))
            < minimum_spacing_m
            and n @ np.asarray(sites[j]["normal_object"]) > 0.5
            for j in selected
        ):
            continue
        selected.append(i)
        if len(selected) == maximum:
            break
    return selected


def inner_wall_faces(
    mesh,
    world_from_object,
    center_xy_m,
    *,
    maximum_radius_m=0.05,
    maximum_abs_vertical_normal=0.6,
    minimum_inward_cosine=0.7,
):
    """Select potential expansion contacts facing a declared opening center.

    Geometry-only eligibility: it does not establish visibility, patch extent,
    insertion access or lift forces. The center and transform must use one world.
    """
    pose, center = np.asarray(world_from_object, float), np.asarray(center_xy_m, float)
    if (
        pose.shape != (4, 4)
        or center.shape != (2,)
        or not np.isfinite(pose).all()
        or not np.isfinite(center).all()
        or not np.allclose(pose[3], [0, 0, 0, 1])
        or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-6)
        or not np.isclose(np.linalg.det(pose[:3, :3]), 1)
        or not 0 < maximum_radius_m <= 0.2
        or not 0 <= maximum_abs_vertical_normal < 1
        or not 0 < minimum_inward_cosine <= 1
    ):
        raise ValueError(
            "Rigid metric world and bounded opening-wall parameters required"
        )
    points = mesh.triangles_center @ pose[:3, :3].T + pose[:3, 3]
    normals = mesh.face_normals @ pose[:3, :3].T
    radial = points[:, :2] - center
    radius = np.linalg.norm(radial, axis=1)
    horizontal = np.linalg.norm(normals[:, :2], axis=1)
    cosine = -np.sum(radial * normals[:, :2], axis=1) / np.maximum(
        radius * horizontal, 1e-12
    )
    return (
        (radius > 1e-6)
        & (radius <= maximum_radius_m)
        & (abs(normals[:, 2]) <= maximum_abs_vertical_normal)
        & (cosine >= minimum_inward_cosine)
    )


def region_masks(mesh, world_from_object, regions):
    """Disjoint bounded surface regions with explicit normal hypotheses.

    Region boxes and normal directions are geometric proposals in declared rigid
    frames. Neither their normals nor the model COM are camera measurements.
    """
    world = np.asarray(world_from_object, float)

    def rigid(pose):
        return (
            pose.shape == (4, 4)
            and np.isfinite(pose).all()
            and np.allclose(pose[3], [0, 0, 0, 1])
            and np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-6)
            and np.isclose(np.linalg.det(pose[:3, :3]), 1)
        )

    if not rigid(world) or not 2 <= len(regions) <= 4:
        raise ValueError("Two to four regions in a rigid metric world required")
    points = mesh.triangles_center @ world[:3, :3].T + world[:3, 3]
    normals = mesh.face_normals @ world[:3, :3].T
    masks = {}
    for region in regions:
        if set(region) != {
            "name",
            "world_from_region",
            "minimum_m",
            "maximum_m",
            "normal_direction",
            "minimum_normal_cosine",
            "maximum_abs_vertical_normal",
        }:
            raise ValueError(
                "Explicit region frame, bounds and normal hypotheses required"
            )
        pose = np.asarray(region["world_from_region"], float)
        lo, hi, direction = [
            np.asarray(region[k], float)
            for k in ("minimum_m", "maximum_m", "normal_direction")
        ]
        cosine, vertical = (
            region["minimum_normal_cosine"],
            region["maximum_abs_vertical_normal"],
        )
        name = region["name"]
        if (
            not isinstance(name, str)
            or not name
            or name in masks
            or not rigid(pose)
            or any(
                v.shape != (3,) or not np.isfinite(v).all() for v in (lo, hi, direction)
            )
            or np.any(lo >= hi)
            or np.max(hi - lo) > 0.5
            or not np.isclose(np.linalg.norm(direction), 1)
            or not np.isfinite([cosine, vertical]).all()
            or not 0 < cosine <= 1
            or not 0 <= vertical < 1
        ):
            raise ValueError(
                "Distinct bounded region boxes and unit normal directions required"
            )
        local = (points - pose[:3, 3]) @ pose[:3, :3]
        local_normals = normals @ pose[:3, :3]
        mask = (
            np.all((local >= lo) & (local <= hi), axis=1)
            & (local_normals @ direction >= cosine)
            & (abs(normals[:, 2]) <= vertical)
        )
        if any(np.any(mask & previous) for previous in masks.values()):
            raise ValueError("Surface region hypotheses must be disjoint")
        masks[name] = mask
    return masks


def sample_regions(
    mesh, world_from_object, regions, *, count=192, normal_distance_m=0.01
):
    masks = region_masks(mesh, world_from_object, regions)
    if (
        type(count) is not int
        or not 2 * len(regions) <= count <= 512
        or count % len(regions)
    ):
        raise ValueError("Equal bounded sample allocation per region required")
    sites = []
    for name, mask in masks.items():
        selected = sample_sites(
            mesh,
            count=count // len(regions),
            normal_distance_m=normal_distance_m,
            eligible_faces=mask,
        )
        for site in selected:
            site.update(site_id=len(sites), region=name)
            sites.append(site)
    return sites
