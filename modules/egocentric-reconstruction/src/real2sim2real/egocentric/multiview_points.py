"""Calibrated appearance correspondences; never anatomical joint labels.

Inputs are synchronized, undistorted pinhole images. Matching uses no hand
model, predicted depth or inferred pose. Ambiguous/textureless patches remain
rejected. Pixel perturbations report sensitivity, not calibrated confidence.
"""

import cv2
import numpy as np

from .camera import intrinsic, rigid, transform


def project_points(xyz, k):
    p = np.asarray(xyz) @ np.asarray(k).T
    return p[..., :2] / p[..., 2:]


def triangulate(points, cameras):
    """DLT in the shared reference camera, with cheirality and reprojection."""
    if len(points) != len(cameras) or len(points) < 2:
        raise ValueError("At least two matched points and cameras required")
    rows, rays = [], []
    for point, (k, view_from_reference) in zip(points, cameras):
        k, t = intrinsic(k), rigid(view_from_reference)
        uv = np.asarray(point, float)
        if uv.shape != (2,) or not np.isfinite(uv).all():
            raise ValueError("Finite image point required")
        # Normalized coordinates avoid weighting cameras by pixel focal length.
        ray = np.linalg.inv(k) @ np.r_[uv, 1]
        rows.extend([ray[0] * t[2] - t[0], ray[1] * t[2] - t[1]])
        direction = t[:3, :3].T @ ray
        rays.append(direction / np.linalg.norm(direction))
    _, _, vt = np.linalg.svd(rows)
    if abs(vt[-1, 3]) < 1e-12:
        raise ValueError("Degenerate triangulation")
    xyz = vt[-1, :3] / vt[-1, 3]
    errors, depths = [], []
    for point, (k, t) in zip(points, cameras):
        p = transform(xyz[None], np.asarray(t))[0]
        depths.append(float(p[2]))
        errors.append(float(np.linalg.norm(project_points(p, k) - point)))
    angles = [
        np.arccos(np.clip(np.dot(a, b), -1, 1))
        for i, a in enumerate(rays)
        for b in rays[i + 1 :]
    ]
    return dict(
        xyz_reference_m=xyz.tolist(),
        reprojection_errors_px=errors,
        positive_depth=bool(min(depths) > 0),
        max_ray_angle_deg=float(np.rad2deg(max(angles))),
    )


def pixel_sensitivity(points, cameras, uncertainty_px=1.0):
    """Worst coordinate displacement from one-at-a-time ±pixel perturbations."""
    if not np.isfinite(uncertainty_px) or uncertainty_px <= 0:
        raise ValueError("Positive finite pixel uncertainty required")
    base = np.asarray(triangulate(points, cameras)["xyz_reference_m"])
    alternatives = []
    for i in range(len(points)):
        for axis in range(2):
            for sign in (-1, 1):
                p = np.asarray(points, float).copy()
                p[i, axis] += sign * uncertainty_px
                alternatives.append(triangulate(p, cameras)["xyz_reference_m"])
    delta = np.asarray(alternatives) - base
    return dict(
        perturbation_px=float(uncertainty_px),
        max_displacement_m=float(np.linalg.norm(delta, axis=1).max()),
        max_depth_change_m=float(np.abs(delta[:, 2]).max()),
        interpretation="sensitivity_only_not_confidence_interval",
    )


def _gray(image):
    x = np.asarray(image)
    if x.ndim == 3 and x.shape[2] == 3:
        x = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("Finite grayscale or RGB image required")
    return x.astype(np.float32)


def _patches(gray, centers, radius):
    offsets = np.stack(
        np.meshgrid(np.arange(-radius, radius + 1), np.arange(-radius, radius + 1)), -1
    )
    xy = np.asarray(centers)[:, None, None] + offsets
    return cv2.remap(
        gray,
        xy[..., 0].astype(np.float32).reshape(-1, 2 * radius + 1),
        xy[..., 1].astype(np.float32).reshape(-1, 2 * radius + 1),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    ).reshape(len(centers), -1)


def epipolar_match(
    source,
    destination,
    point,
    source_k,
    destination_k,
    destination_from_source,
    *,
    min_depth_m=0.2,
    max_depth_m=1.5,
    radius=6,
    min_ncc=0.85,
    min_margin=0.08,
    min_texture=3.0,
):
    """Search the full configured depth interval, independent of any 3D prior."""
    k1, k2, t = (
        intrinsic(source_k),
        intrinsic(destination_k),
        rigid(destination_from_source),
    )
    if (
        not 0 < min_depth_m < max_depth_m
        or type(radius) is not int
        or not 2 <= radius <= 15
    ):
        raise ValueError("Invalid metric search interval or patch radius")
    if not -1 <= min_ncc <= 1 or not 0 <= min_margin <= 2 or min_texture <= 0:
        raise ValueError("Invalid patch acceptance limits")
    a, b, uv = _gray(source), _gray(destination), np.asarray(point, float)
    if uv.shape != (2,) or not np.isfinite(uv).all():
        raise ValueError("Finite source pixel required")
    row = dict(accepted=False, reasons=[], anatomical_identity_verified=False)
    if not (
        radius <= uv[0] < a.shape[1] - radius - 1
        and radius <= uv[1] < a.shape[0] - radius - 1
    ):
        return dict(row, reasons=["source_patch_outside_image"])
    ray = np.linalg.inv(k1) @ np.r_[uv, 1]
    ends3 = transform(ray[None] * np.array([min_depth_m, max_depth_m])[:, None], t)
    if (ends3[:, 2] <= 0).any():
        return dict(row, reasons=["search_behind_camera"])
    ends = project_points(ends3, k2)
    length = np.linalg.norm(ends[1] - ends[0])
    if length < 4 or length > 1000:
        return dict(row, reasons=["unsupported_epipolar_search_extent"])
    centers = np.linspace(ends[0], ends[1], int(np.ceil(length * 2)) + 1)
    inside = (
        (centers[:, 0] >= radius)
        & (centers[:, 0] < b.shape[1] - radius - 1)
        & (centers[:, 1] >= radius)
        & (centers[:, 1] < b.shape[0] - radius - 1)
    )
    centers = centers[inside]
    if len(centers) < 9:
        return dict(row, reasons=["insufficient_search_pixels"])
    template = _patches(a, uv[None], radius)[0]
    row["texture_std"] = float(template.std())
    if template.std() < min_texture:
        return dict(row, reasons=["low_texture"])
    template -= template.mean()
    patches = _patches(b, centers, radius)
    patches -= patches.mean(axis=1, keepdims=True)
    scores = (
        patches
        @ template
        / np.maximum(np.linalg.norm(patches, axis=1) * np.linalg.norm(template), 1e-9)
    )
    best = int(scores.argmax())
    separated = np.linalg.norm(centers - centers[best], axis=1) >= 3
    margin = float(scores[best] - scores[separated].max()) if separated.any() else 0.0
    row.update(
        match_px=centers[best].tolist(),
        ncc=float(scores[best]),
        uniqueness_margin=margin,
    )
    if scores[best] < min_ncc:
        row["reasons"].append("weak_photometric_match")
    if margin < min_margin:
        row["reasons"].append("ambiguous_photometric_match")
    if best in (0, len(centers) - 1):
        row["reasons"].append("search_boundary")
    row["accepted"] = not row["reasons"]
    return row


def reciprocal_match(
    source,
    destination,
    point,
    source_k,
    destination_k,
    destination_from_source,
    **options,
):
    """Require unique forward/reverse appearance and ≤2 px cycle error."""
    row = epipolar_match(
        source,
        destination,
        point,
        source_k,
        destination_k,
        destination_from_source,
        **options,
    )
    if not row["accepted"]:
        return row
    reverse = epipolar_match(
        destination,
        source,
        row["match_px"],
        destination_k,
        source_k,
        np.linalg.inv(destination_from_source),
        **options,
    )
    row["reverse"] = reverse
    row["cycle_error_px"] = (
        float(np.linalg.norm(np.asarray(reverse["match_px"]) - point))
        if "match_px" in reverse
        else None
    )
    if not reverse["accepted"]:
        row["reasons"].append("reverse_match_rejected")
    if row["cycle_error_px"] is None or row["cycle_error_px"] > 2:
        row["reasons"].append("cycle_inconsistent")
    row["accepted"] = not row["reasons"]
    return row
