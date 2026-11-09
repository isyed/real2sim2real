"""Fit a measured support plane and apply one rigid upright scene transform.

Up is inferred from the assumption that the selected support is level. This is
not an IMU gravity measurement. Unobserved support geometry is not recovered.
"""

from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import ConvexHull

from real2sim2real.artifacts import digest, write_json
from .camera import transform, unproject, rigid
from .interaction_io import Reconstruction, fresh_directory


def fit_plane(points, *, threshold=0.008, trials=300, seed=0):
    points = np.asarray(points, dtype=float)
    points = points[np.isfinite(points).all(1)]
    if len(points) < 100:
        raise ValueError("Too few measured support points")
    rng = np.random.default_rng(seed)
    sample = points[rng.choice(len(points), min(6000, len(points)), replace=False)]
    best, score = None, 0
    for _ in range(trials):
        p = sample[rng.choice(len(sample), 3, replace=False)]
        n = np.cross(p[1] - p[0], p[2] - p[0])
        if np.linalg.norm(n) < 1e-8:
            continue
        n /= np.linalg.norm(n)
        d = -n @ p[0]
        current = int((np.abs(sample @ n + d) < threshold).sum())
        if current > score:
            best, score = (n, d), current
    if best is None or score < 100:
        raise ValueError("Support plane is not observable")
    n, d = best
    for _ in range(5):
        inliers = np.abs(points @ n + d) < threshold
        p = points[inliers]
        centre = np.mean(p, axis=0)
        _, singular, vh = np.linalg.svd(p - centre, full_matrices=False)
        if singular[1] < 0.05:
            raise ValueError("Support observations are collinear")
        n = vh[-1]
        d = -n @ centre
    return n, float(d), np.abs(points @ n + d) < threshold


def upright_transform(normal, offset, origin, camera_position=(0, 0, 0)):
    n = np.asarray(normal, dtype=float)
    if not np.isclose(np.linalg.norm(n), 1):
        raise ValueError("Plane normal must be unit length")
    centre = np.asarray(origin, dtype=float)
    centre = centre - n * (n @ centre + offset)
    if n @ (np.asarray(camera_position) - centre) < 0:
        n = -n
    x = np.array([1.0, 0, 0]) - n * n[0]
    if np.linalg.norm(x) < 0.1:
        x = np.array([0.0, 0, 1]) - n * n[2]
    x /= np.linalg.norm(x)
    y = np.cross(n, x)
    result = np.eye(4)
    result[:3, :3] = np.stack([x, y, n])
    result[:3, 3] = -result[:3, :3] @ centre
    return rigid(result)


def support_mask(rgb, options):
    """Explicit image-only selection: ROI/polygon plus optional HSV interval."""
    h, w = rgb.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    polygon = options.get("polygon_normalized")
    if polygon is not None:
        vertices = np.rint(np.asarray(polygon) * [w - 1, h - 1]).astype(np.int32)
        cv2.fillPoly(mask, [vertices], 1)
    else:
        x0, y0, x1, y1 = options["roi_normalized"]
        mask[int(y0 * h) : int(y1 * h), int(x0 * w) : int(x1 * w)] = 1
    if "hsv_lower" in options:
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        mask &= (
            cv2.inRange(
                hsv,
                np.array(options["hsv_lower"], np.uint8),
                np.array(options["hsv_upper"], np.uint8),
            )
            > 0
        )
    if options.get("largest_component", False):
        _, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        if len(stats) > 1:
            mask = (labels == (1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(
                np.uint8
            )
    radius = options.get("selection_erosion_pixels", 3)
    if radius:
        mask = cv2.erode(mask, np.ones((2 * radius + 1, 2 * radius + 1), np.uint8))
    return mask.astype(bool)


def observed_points(source, index, options):
    depth = source.depth(index)
    mask = support_mask(source.rgb(index), options)
    for identifier in options.get("exclude_mask_ids", [1, 2, 3]):
        # Avoid mixed foreground/background disparity at mask boundaries.
        radius = options.get("foreground_margin_pixels", 10)
        foreground = cv2.dilate(
            source.mask(index, identifier).astype(np.uint8),
            np.ones((2 * radius + 1, 2 * radius + 1), np.uint8),
        )
        mask &= foreground == 0
    mask &= (
        np.isfinite(depth)
        & (depth > options.get("min_depth_m", 0.2))
        & (depth < options.get("max_depth_m", 1.5))
    )
    xyz = unproject(depth, source.k)[mask]
    if len(xyz) > 12000:
        xyz = xyz[np.random.default_rng(index).choice(len(xyz), 12000, replace=False)]
    return transform(xyz, source.world_from_camera[index]), mask


def prism_mesh(boundary, thickness):
    """Closed convex collision slab; its top surface is exactly z=0."""
    import trimesh

    boundary = np.asarray(boundary)
    n = len(boundary)
    vertices = np.concatenate(
        [np.c_[boundary, np.zeros(n)], np.c_[boundary, np.full(n, -thickness)]]
    )
    faces = []
    for i in range(1, n - 1):
        faces += [[0, i, i + 1], [n, n + i + 1, n + i]]
    for i in range(n):
        j = (i + 1) % n
        faces += [[i, n + i, n + j], [i, n + j, j]]
    mesh = trimesh.Trimesh(vertices, faces, process=True)
    mesh.fix_normals()
    return mesh


def reconstruct_support(config):
    source = Reconstruction(config["reconstruction"], config.get("frame_count"))
    out = fresh_directory(Path(config["output"]) / "support")
    options = config["support"]
    fit_frames, holdout = options["fit_frames"], options["validation_frames"]
    if set(fit_frames) & set(holdout) or any(
        i < 0 or i >= source.count for i in fit_frames + holdout
    ):
        raise ValueError(
            "Support fit and validation frames must be disjoint and in range"
        )
    threshold = options.get("inlier_threshold_m", 0.008)
    points = np.concatenate(
        [observed_points(source, i, options)[0] for i in fit_frames]
    )
    normal, offset, inliers = fit_plane(points, threshold=threshold)
    upright = upright_transform(
        normal,
        offset,
        np.median(points[inliers], axis=0),
        source.world_from_camera[0, :3, 3],
    )
    xy = transform(points[inliers], upright)[:, :2]
    # Remove isolated extremal depth samples before estimating the observed hull.
    lo, hi = np.quantile(xy, [0.005, 0.995], axis=0)
    xy = xy[((xy >= lo) & (xy <= hi)).all(1)]
    boundary = xy[ConvexHull(xy).vertices]
    validation = []
    for i in holdout:
        p, mask = observed_points(source, i, options)
        residual = np.abs(transform(p, upright)[:, 2])
        row = dict(
            frame=int(i),
            points=len(p),
            median_m=float(np.median(residual)) if len(p) else None,
            p90_m=float(np.quantile(residual, 0.9)) if len(p) else None,
            within_2cm_fraction=float((residual < 0.02).mean()) if len(p) else 0,
        )
        row["accepted"] = bool(
            len(p) >= 100
            and row["median_m"] <= options.get("max_validation_median_m", 0.01)
            and row["within_2cm_fraction"]
            >= options.get("minimum_validation_consensus", 0.75)
        )
        if "max_validation_p90_m" in options:
            row["accepted"] &= (
                row["p90_m"] is not None
                and row["p90_m"] <= options["max_validation_p90_m"]
            )
        validation.append(row)
        rgb = source.rgb(i).copy()
        rgb[mask] = (0.45 * rgb[mask] + 0.55 * np.array([40, 220, 60])).astype(np.uint8)
        cv2.imwrite(
            str(out / f"selection_{i:06d}.jpg"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        )
    mesh = prism_mesh(boundary, options.get("collision_thickness_m", 0.03))
    mesh.export(out / "tabletop.glb")
    quality = dict(
        schema="real2sim2real.support/1",
        units="metres",
        up_axis="+Z",
        upright_from_world=upright.tolist(),
        observed_boundary_xy=boundary.tolist(),
        assumption="selected tabletop is level; normal oriented toward first camera",
        boundary_kind="convex hull of observed support; occluded extent is unverified",
        collision_thickness_m=options.get("collision_thickness_m", 0.03),
        fit_frames=fit_frames,
        fit_points=len(points),
        fit_inlier_fraction=float(inliers.mean()),
        fit_median_m=float(
            np.median(np.abs(transform(points[inliers], upright)[:, 2]))
        ),
        validation=validation,
        accepted=all(r["accepted"] for r in validation) and bool(validation),
        source_geometry_sha256=digest(source.root / "geometry.npz"),
        ground_truth_used=False,
    )
    write_json(out / "support.json", quality)
    np.savez_compressed(
        out / "scene.npz",
        upright_from_world=upright,
        world_from_camera=upright @ source.world_from_camera,
        world_from_object=upright @ source.world_from_object,
        intrinsics=source.k,
        frame_ids=source.frame_ids,
        timestamps_s=source.times,
    )
    if not quality["accepted"]:
        raise ValueError("Support validation failed; rejected hypothesis retained")
    return quality
