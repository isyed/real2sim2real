"""Local occupied-cell contact hypotheses, separate from planning exclusions."""

import numpy as np
from scipy.spatial import cKDTree


def occupied_boxes(points_m, *, voxel_m=0.006, padding_m=0.0, max_cells=8000):
    """Cover supplied observations without bridging disconnected occupied cells.

    Cell thickness is a modeling assumption. Empty cells are unobserved, not
    certified free space. No filtering depends on the grasp or object collision.
    """
    p = np.asarray(points_m, float)
    if (
        p.ndim != 2
        or p.shape[1] != 3
        or not len(p)
        or not np.isfinite(p).all()
        or not 0.002 <= voxel_m <= 0.02
        or not 0 <= padding_m <= 0.003
        or not 1 <= max_cells <= 20000
    ):
        raise ValueError("Finite metric observations and bounded voxel budget required")
    keys, counts = np.unique(
        np.floor(p / voxel_m).astype(np.int64), axis=0, return_counts=True
    )
    if len(keys) > max_cells:
        raise ValueError("Occupied-cell budget exceeded")
    return {
        "centers_m": (keys + 0.5) * voxel_m,
        "half_extents_m": np.full((len(keys), 3), voxel_m / 2 + padding_m),
        "counts": counts,
        "voxel_m": voxel_m,
        "padding_m": padding_m,
    }


def coverage_upper_bounds(points_m, boxes, *, neighbors=8):
    """Distance to nearby occupied boxes; conservative upper bound on nearest distance."""
    p = np.asarray(points_m, float)
    centers = np.asarray(boxes["centers_m"], float)
    half = np.asarray(boxes["half_extents_m"], float)
    if (
        p.ndim != 2
        or p.shape[1] != 3
        or not len(p)
        or not np.isfinite(p).all()
        or centers.ndim != 2
        or centers.shape[1] != 3
        or half.shape != centers.shape
        or not len(centers)
        or not np.isfinite([centers, half]).all()
        or (half <= 0).any()
        or not 1 <= neighbors <= 32
    ):
        raise ValueError("Finite metric samples and a bounded neighbor count required")
    _, idx = cKDTree(centers).query(p, k=min(neighbors, len(centers)))
    idx = np.asarray(idx).reshape(len(p), -1)
    distance = np.maximum(abs(p[:, None, :] - centers[idx]) - half[idx], 0)
    return np.linalg.norm(distance, axis=2).min(axis=1)


def apply_contact_roles(tree, rack_geom, boxes, *, planning_only_geoms):
    """Keep exclusions in XML but disable their force contacts in this separate fork.

    Callers must retain an independent planning check for disabled exclusions.
    The original rack is retained as an exclusion; occupied box geoms supply
    physical contact. The supplied tree is modified, never the source file.
    """
    import xml.etree.ElementTree as ET

    world = tree.getroot().find("worldbody")
    if world is None:
        raise ValueError("Worldbody required")
    names = [rack_geom, *planning_only_geoms]
    if len(set(names)) != len(names):
        raise ValueError("Distinct geometry roles required")
    for name in names:
        matches = [g for g in world.findall("geom") if g.get("name") == name]
        if len(matches) != 1:
            raise ValueError("Named world exclusion required")
        g = matches[0]
        g.set("contype", "0")
        g.set("conaffinity", "0")
        g.set("group", "4")
    centers = np.asarray(boxes["centers_m"], float)
    half = np.asarray(boxes["half_extents_m"], float)
    if (
        centers.shape != half.shape
        or centers.ndim != 2
        or centers.shape[1] != 3
        or not len(centers)
        or not np.isfinite([centers, half]).all()
        or (half <= 0).any()
    ):
        raise ValueError("Finite occupied boxes required")
    created = []
    for i, (p, h) in enumerate(zip(centers, half)):
        name = f"{rack_geom}_observed_{i}"
        if any(g.get("name") == name for g in world.findall("geom")):
            raise ValueError("Contact geom already exists")
        ET.SubElement(
            world,
            "geom",
            name=name,
            type="box",
            pos=" ".join(map(str, p)),
            size=" ".join(map(str, h)),
            contype="1",
            conaffinity="1",
            rgba=".75 .75 .7 1",
        )
        created.append(name)
    return {
        "planning_only": names,
        "physical_geoms": created,
        "unobserved_space_certified_free": False,
    }


def box_observation_evidence(
    points_m, center_m, half_extents_m, *, world_from_box_rotation=None
):
    """Measure evidence for one box without equating missing depth with free space.

    Call separately per view and retain fit/validation identities in the caller.
    Observations must already be valid, masked samples in the same metric frame.
    Thickness, segmentation accuracy and occluded surfaces remain assumptions.
    """
    p = np.asarray(points_m, float)
    center, half = np.asarray(center_m, float), np.asarray(half_extents_m, float)
    rotation = (
        np.eye(3)
        if world_from_box_rotation is None
        else np.asarray(world_from_box_rotation, float)
    )
    if (
        p.ndim != 2
        or p.shape[1] != 3
        or center.shape != (3,)
        or half.shape != (3,)
        or not np.isfinite(p).all()
        or not np.isfinite(center).all()
        or not np.isfinite(half).all()
        or np.any(half <= 0)
        or rotation.shape != (3, 3)
        or not np.isfinite(rotation).all()
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7)
        or not np.isclose(np.linalg.det(rotation), 1)
    ):
        raise ValueError(
            "Finite metric points, positive box extents and proper rotation required"
        )
    result = {
        "observation_count": len(p),
        "observations_inside": 0,
        "nearest_observation_distance_to_box_m": None,
        "nearest_observation_m": None,
        "missing_evidence_is_not_free_space": True,
    }
    if len(p):
        distances = np.linalg.norm(
            np.maximum(np.abs((p - center) @ rotation) - half, 0), axis=1
        )
        index = int(np.argmin(distances))
        result.update(
            observations_inside=int(np.sum(distances <= 1e-12)),
            nearest_observation_distance_to_box_m=float(distances[index]),
            nearest_observation_m=p[index].tolist(),
        )
    return result
