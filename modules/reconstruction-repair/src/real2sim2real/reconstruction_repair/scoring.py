"""Before/after diagnostics, explicitly separate from downstream task evaluation."""

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .bundle import arrays_read, asset_path
from .contracts import ReconstructionBundle
from .geometry import (frame_overlap, mesh_audit, object_camera_centers, object_centers,
                       object_world_vertices, points_mesh_distance, reliable_frames, rotation_angle)


@dataclass
class ScoreReport:
    summary: dict
    per_frame: dict


def _stats(value: np.ndarray) -> dict:
    value = np.asarray(value)
    value = value[np.isfinite(value)]
    return {"mean": float(np.mean(value)), "p95": float(np.percentile(value, 95)),
            "max": float(np.max(value)), "min": float(np.min(value))} if len(value) else {
                "mean": None, "p95": None, "max": None, "min": None}


def motion_statistics(g: dict, max_gap_s: float = 0.2) -> dict:
    """Timestamp-aware motion definitions shared by reporting and proposal limiting."""
    n = len(g["frame_ids"])
    centers = object_centers(g)
    dt = np.diff(g["timestamps_s"])
    adjacent = dt <= max_gap_s
    velocity = np.diff(centers, axis=0) / dt[:, None]
    acceleration = np.diff(velocity, axis=0) / ((dt[:-1] + dt[1:]) * 0.5)[:, None]
    triple = adjacent[:-1] & adjacent[1:]
    angles = np.rad2deg(rotation_angle(g["object_rotation"][1:], g["object_rotation"][:-1])) if n > 1 else np.array([])
    return {
        "center_speed_m_s": _stats(np.linalg.norm(velocity[adjacent], axis=1)),
        "center_acceleration_m_s2": _stats(np.linalg.norm(acceleration[triple], axis=1)),
        "rotation_step_deg": _stats(angles[adjacent]),
        "angular_speed_deg_s": _stats((angles / dt)[adjacent]),
    }


def geometry_scores(bundle: ReconstructionBundle, g: dict | None = None, max_gap_s: float = 0.2) -> ScoreReport:
    g = bundle.geometry if g is None else g
    o = bundle.observations
    n = len(g["frame_ids"])
    valid = reliable_frames(o, n)
    iou = np.full(n, np.nan)
    near_plane = np.zeros(n, bool)
    for i in range(n):
        w = g["world_to_camera"][i]
        camera = object_world_vertices(g, i) @ w[:3, :3].T + w[:3, 3]
        near_plane[i] = np.any(camera[np.unique(g["faces"]), 2] <= 1e-5)
        if valid[i] and not near_plane[i]:
            iou[i] = frame_overlap(g, o, i)
    motion = motion_statistics(g, max_gap_s)
    gaps = np.full((n, 0), np.nan)
    if "hand_points_world" in o:
        gaps = np.stack([points_mesh_distance(o["hand_points_world"][i], object_world_vertices(g, i), g["faces"])
                         for i in range(n)])
    semantic = np.full(n, np.nan)
    evidence = bundle.manifest.get("evidence", {})
    if "axis_facing" in o and "semantic_axis_local" in evidence:
        axis = np.asarray(evidence["semantic_axis_local"], dtype=float)
        axis /= np.linalg.norm(axis)
        direction = np.einsum("tij,j->ti", g["object_rotation"], axis)
        direction = np.einsum("tij,tj->ti", g["world_to_camera"][:, :3, :3], direction)
        to_camera = -object_camera_centers(g)
        to_camera /= np.maximum(np.linalg.norm(to_camera, axis=1, keepdims=True), 1e-12)
        dot = np.einsum("ij,ij->i", direction, to_camera)
        labelled = o["axis_facing"] != 0
        semantic[labelled] = (dot[labelled] * o["axis_facing"][labelled] > 0.1).astype(float)
    topology = mesh_audit(g)
    scale_error = None
    if "scale" in evidence:
        s = evidence["scale"]
        scale_error = abs(topology["local_extent_m"][s["axis"]] - s["length_m"])
    summary = {
        "frames": n, "image_reviewed_frames": int(np.isfinite(iou).sum()),
        "near_plane_invalid_frames": int(near_plane.sum()),
        "max_rotation_orthogonality_error": float(np.max(np.abs(g["object_rotation"] @ g["object_rotation"].swapaxes(-1, -2) - np.eye(3)))),
        "max_rotation_determinant_error": float(np.max(np.abs(np.linalg.det(g["object_rotation"]) - 1))),
        "silhouette_iou": _stats(iou), "mesh": topology,
        **motion,
        "hand_surface_gap_m": _stats(gaps),
        "annotated_contact_gap_m": _stats(gaps[o["contact_active"].astype(bool)]) if gaps.size else _stats([]),
        "semantic_facing_agreement": _stats(semantic), "scale_absolute_error_m": scale_error,
    }
    return ScoreReport(summary, {"frame_ids": g["frame_ids"], "iou": iou,
                                "near_plane_invalid": near_plane, "hand_gap_m": gaps,
                                "semantic_agreement": semantic, "camera_center_m": object_camera_centers(g)})


def scores_compare(before: ScoreReport, after: ScoreReport) -> dict:
    """Keep raw metrics separate: no invented composite quality score."""
    deltas = {}
    for key in before.summary:
        a, b = before.summary[key], after.summary[key]
        if isinstance(a, dict) and "mean" in a:
            deltas[key] = {stat: (b[stat] - a[stat]) if a[stat] is not None and b[stat] is not None else None
                           for stat in a}
        elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
            deltas[key] = b - a
    depth_drift = after.per_frame["camera_center_m"][:, 2] - before.per_frame["camera_center_m"][:, 2]
    return {"before": before.summary, "after": after.summary, "after_minus_before": deltas,
            "camera_center_depth_change_m": _stats(np.abs(depth_drift)),
            "interpretation": {
                "higher_is_better": ["silhouette_iou", "semantic_facing_agreement"],
                "lower_is_better_given_evidence": ["scale_absolute_error_m"],
                "context_required": ["motion: lower can erase real motion", "gap: lower can mean penetration",
                                     "silhouette: cannot certify depth, orientation, identity or contact"],
                "official_scores": "Unavailable without the official ground truth and evaluator; these are diagnostics.",
                "human": "Human payload is preserved. Object-only repairs do not directly improve CD-H.",
            }}


def _sample_mesh(g: dict, count: int, seed: int) -> np.ndarray:
    v, f = g["vertices"] * g["object_scale"], g["faces"]
    tri = v[f]
    area = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    if area.sum() <= 0:
        raise ValueError("Reference scoring needs a nondegenerate mesh")
    rng = np.random.default_rng(seed)
    tri = tri[rng.choice(len(f), size=count, p=area / area.sum())]
    uv = rng.random((count, 2))
    uv[uv.sum(axis=1) > 1] = 1 - uv[uv.sum(axis=1) > 1]
    return tri[:, 0] + uv[:, :1] * (tri[:, 1] - tri[:, 0]) + uv[:, 1:] * (tri[:, 2] - tri[:, 0])


def _chamfer(a: np.ndarray, b: np.ndarray) -> float:
    return float((cKDTree(a).query(b)[0].mean() + cKDTree(b).query(a)[0].mean()) * 50)


def reference_scores(prediction: ReconstructionBundle, reference: ReconstructionBundle,
                     samples: int = 2048, seed: int = 0) -> dict:
    """Reference is for evaluation only, never an input to repair/selection."""
    p, r = prediction.geometry, reference.geometry
    if not np.array_equal(p["frame_ids"], r["frame_ids"]) or not np.allclose(p["timestamps_s"], r["timestamps_s"], atol=1e-8):
        raise ValueError("Reference frame IDs and timestamps must match exactly")
    if prediction.manifest.get("world_frame_id") != reference.manifest.get("world_frame_id") or not prediction.manifest.get("world_frame_id"):
        raise ValueError("Reference scoring requires the same explicit world_frame_id")
    if samples < 32:
        raise ValueError("Use at least 32 reference surface samples")
    ps, rs = _sample_mesh(p, samples, seed), _sample_mesh(r, samples, seed)
    cd = [_chamfer(ps @ p["object_rotation"][i].T + p["object_translation"][i],
                   rs @ r["object_rotation"][i].T + r["object_translation"][i]) for i in range(len(p["frame_ids"]))]
    result = {
        "label": "Reference diagnostics, not the official evaluator or leaderboard score",
        "samples_per_object_surface": samples, "seed": seed,
        "symmetric_surface_chamfer_object_cm": _stats(cd), "per_frame_object_cm": cd,
        "object_center_error_cm": _stats(np.linalg.norm(object_centers(p) - object_centers(r), axis=1) * 100),
        "rotation_error_deg_unadjusted_for_symmetry": _stats(np.rad2deg(rotation_angle(p["object_rotation"], r["object_rotation"]))),
        "human_vertex_chamfer_cm": None,
        "penetration_cm": None,
    }
    if "human" in prediction.manifest["assets"] and "human" in reference.manifest["assets"]:
        hp = arrays_read(asset_path(prediction.root, prediction.manifest["assets"]["human"]))
        hr = arrays_read(asset_path(reference.root, reference.manifest["assets"]["human"]))
        if "vertices_world" in hp and "vertices_world" in hr:
            a, b = hp["vertices_world"], hr["vertices_world"]
            if a.ndim != 3 or b.ndim != 3 or a.shape[0] != len(cd) or b.shape[0] != len(cd) or a.shape[2] != 3 or b.shape[2] != 3:
                raise ValueError("Decoded human vertices_world must be [T,V,3] in meters")
            result["human_vertex_chamfer_cm"] = _stats([_chamfer(x, y) for x, y in zip(a, b)])
    return result
