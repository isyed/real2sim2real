"""Fit an explicit metric floor from reviewed RGB regions and predicted depth.

The caller supplies native image dimensions and independently reviewed XYXY
regions. Foreground masks are excluded. Plane validity is checked separately
from object resting contact, which this module never infers.
"""
from dataclasses import dataclass, asdict
import numpy as np


@dataclass(frozen=True)
class PlaneGates:
    seed: int = 0
    ransac_trials: int = 256
    sample_limit: int = 12000
    distance_m: float = .02
    min_points: int = 500
    min_region_points: int = 100
    min_inlier_fraction: float = .70
    min_region_inlier_fraction: float = .60
    max_inlier_rms_m: float = .0125
    min_plane_span_m: float = .15
    min_camera_height_m: float = .20
    max_camera_height_m: float = 3.0
    min_up_component: float = .25


class PlaneFitError(ValueError):
    def __init__(self, message, diagnostics):
        super().__init__(message)
        self.diagnostics = {**diagnostics, "passed": False, "failure": message}


def floor_points(depth, K, human_mask, object_mask, selection):
    depth, K = np.asarray(depth), np.asarray(K, dtype=np.float64)
    width, height = selection["image_size"]
    if (depth.shape != (height, width) or human_mask.shape != depth.shape or object_mask.shape != depth.shape):
        raise ValueError("Depth/mask/RGB dimensions differ; implicit intrinsics resizing is forbidden")
    if K.shape != (3, 3) or not np.isfinite(K).all() or np.linalg.det(K) <= 0:
        raise ValueError("Invalid pixel intrinsics")
    owner = np.full(depth.shape, -1, dtype=np.int16)
    points, labels, diagnostics = [], [], []
    for index, (x0, y0, x1, y1) in enumerate(selection["rectangles_xyxy"]):
        if np.any(owner[y0:y1, x0:x1] != -1):
            raise ValueError("Floor ROIs must not overlap")
        owner[y0:y1, x0:x1] = index
        roi = owner == index
        excluded = roi & (human_mask | object_mask)
        valid = roi & ~excluded & np.isfinite(depth) & (depth > 0)
        v, u = np.nonzero(valid)
        rays = np.column_stack((u, v, np.ones(len(u)))) @ np.linalg.inv(K).T
        xyz = rays * depth[v, u, None]
        points.append(xyz)
        labels.append(np.full(len(xyz), index, dtype=np.int32))
        diagnostics.append({"region": index, "roi_pixels": int(roi.sum()),
                            "excluded_mask_pixels": int(excluded.sum()), "valid_depth_pixels": len(xyz)})
    return np.concatenate(points), np.concatenate(labels), diagnostics


def fit_floor_plane(points, regions, *, gates=PlaneGates()):
    """Deterministic RANSAC and orthogonal least-squares refinement in CV metres."""
    points, regions = np.asarray(points, dtype=np.float64), np.asarray(regions)
    diagnostics = {"method": "deterministic_ransac_then_orthogonal_plane_fit", "gates": asdict(gates),
                   "point_count": len(points), "passed": False}
    if (points.ndim != 2 or points.shape[1:] != (3,) or not np.isfinite(points).all()
            or regions.shape != (len(points),) or regions.dtype.kind not in "iu"):
        raise PlaneFitError("Invalid floor point arrays", diagnostics)
    ids = np.unique(regions)
    if len(points) < gates.min_points or len(ids) < 2 or any(np.sum(regions == i) < gates.min_region_points for i in ids):
        raise PlaneFitError("Insufficient valid floor points in both independent ROIs", diagnostics)
    rng = np.random.default_rng(gates.seed)
    sampled = points[rng.choice(len(points), min(len(points), gates.sample_limit), replace=False)]
    best, best_count, best_error = None, -1, float("inf")
    for _ in range(gates.ransac_trials):
        a, b, c = sampled[rng.choice(len(sampled), 3, replace=False)]
        normal = np.cross(b - a, c - a)
        length = np.linalg.norm(normal)
        if length < 1e-10:
            continue
        normal /= length
        offset = -float(normal @ a)
        residual = np.abs(np.einsum("ij,j->i", sampled, normal) + offset)
        keep = residual <= gates.distance_m
        count = int(keep.sum())
        error = float(np.mean(residual[keep] ** 2)) if count else float("inf")
        if count > best_count or count == best_count and error < best_error:
            best, best_count, best_error = np.r_[normal, offset], count, error
    if best is None:
        raise PlaneFitError("Floor points do not define a plane", diagnostics)
    keep = np.abs(np.einsum("ij,j->i", points, best[:3]) + best[3]) <= gates.distance_m
    singular = None
    for _ in range(3):
        if keep.sum() < 3:
            raise PlaneFitError("Insufficient plane inliers", diagnostics)
        inliers = points[keep]
        center = inliers.mean(axis=0)
        _, singular, vectors = np.linalg.svd(inliers - center, full_matrices=False)
        normal = vectors[-1]
        best = np.r_[normal, -normal @ center]
        keep = np.abs(np.einsum("ij,j->i", points, best[:3]) + best[3]) <= gates.distance_m
    if best[3] < 0:
        best = -best  # camera origin lies above the selected floor
    residual = np.abs(np.einsum("ij,j->i", points, best[:3]) + best[3])
    if not keep.any():
        raise PlaneFitError("No final plane inliers", diagnostics)
    per_region = [{"region": int(i), "points": int(np.sum(regions == i)),
                   "inliers": int(np.sum(keep & (regions == i))),
                   "inlier_fraction": float(np.mean(keep[regions == i]))} for i in ids]
    span = float(singular[1] / np.sqrt(len(inliers)))
    rms = float(np.sqrt(np.mean(residual[keep] ** 2)))
    diagnostics.update(plane_camera_cv=best.tolist(), inlier_count=int(keep.sum()),
        inlier_fraction=float(keep.mean()), inlier_rms_m=rms,
        inlier_p95_m=float(np.percentile(residual[keep], 95)), all_points_rms_m=float(np.sqrt(np.mean(residual ** 2))),
        secondary_axis_std_m=span, camera_height_m=float(best[3]), regions=per_region)
    failures = []
    if keep.mean() < gates.min_inlier_fraction:
        failures.append("overall inlier fraction")
    if any(r["inlier_fraction"] < gates.min_region_inlier_fraction or r["inliers"] < gates.min_region_points for r in per_region):
        failures.append("per-region inlier coverage")
    if rms > gates.max_inlier_rms_m:
        failures.append("plane RMS")
    if span < gates.min_plane_span_m:
        failures.append("two-dimensional spatial support")
    if not gates.min_camera_height_m <= best[3] <= gates.max_camera_height_m:
        failures.append("camera height")
    if -best[1] < gates.min_up_component:
        failures.append("floor normal direction")
    if failures:
        raise PlaneFitError("Floor-fit gates failed: " + ", ".join(failures), diagnostics)
    return best, {**diagnostics, "passed": True}
