"""Partial metric surfaces from stereo, without completing hidden geometry.

This CPU screen precedes learned mesh generation or robot fitting. A sampled
free-space check is not a swept robot collision certificate. Dense SGBM points
are correlated observations, not independent contact labels.
"""

from __future__ import annotations

import cv2
import numpy as np

from .camera import rigid, transform


def reciprocal_disparity_mask(left, right, *, maximum_error_px=1.0):
    """Right map stores positive disparity at right-image coordinates."""
    a, b = np.asarray(left, float), np.asarray(right, float)
    if (
        a.ndim != 2
        or a.shape != b.shape
        or not np.isfinite(maximum_error_px)
        or maximum_error_px <= 0
    ):
        raise ValueError("Matching disparity grids and positive error bound required")
    y, x = np.indices(a.shape)
    valid = np.isfinite(a) & (a > 0)
    rx = np.rint(x - np.where(valid, a, 0)).astype(int)
    inside = (rx >= 0) & (rx < a.shape[1])
    reverse = b[y, np.clip(rx, 0, a.shape[1] - 1)]
    error = np.abs(a - reverse)
    valid &= inside & np.isfinite(reverse) & (reverse > 0) & (error <= maximum_error_px)
    return valid, error


def stereo_observations(
    model,
    left,
    right,
    *,
    minimum_texture_std=5.0,
    maximum_roundtrip_px=1.0,
    depth_range_m=(0.15, 2.0),
):
    """Retain rectified pixels, raw/filtered depth and disparity sensitivity."""
    if not (
        np.isfinite(minimum_texture_std)
        and minimum_texture_std > 0
        and 0 < depth_range_m[0] < depth_range_m[1]
    ):
        raise ValueError("Positive texture threshold and depth interval required")
    rect = [
        cv2.remap(
            cv2.cvtColor(im, cv2.COLOR_RGB2GRAY), *model.maps[i], cv2.INTER_LINEAR
        )
        for i, im in enumerate((left, right))
    ]
    disparity = model.matcher.compute(*rect).astype(np.float32) / 16
    reverse = (
        model.matcher.compute(
            np.ascontiguousarray(rect[1][:, ::-1]),
            np.ascontiguousarray(rect[0][:, ::-1]),
        )[:, ::-1].astype(np.float32)
        / 16
    )
    reciprocal, error = reciprocal_disparity_mask(
        disparity, reverse, maximum_error_px=maximum_roundtrip_px
    )
    gray = rect[0].astype(np.float32)
    variance = cv2.blur(gray * gray, (7, 7)) - cv2.blur(gray, (7, 7)) ** 2
    texture = np.sqrt(np.maximum(variance, 0))
    points = cv2.reprojectImageTo3D(disparity, model.Q)
    raw = (disparity > 0) & np.isfinite(points).all(-1)
    raw &= (points[..., 2] >= depth_range_m[0]) & (points[..., 2] <= depth_range_m[1])
    valid = raw & reciprocal & (texture >= minimum_texture_std)
    # First-order sensitivity to 0.5 px, not a confidence interval and not
    # calibration or systematic correspondence uncertainty.
    sensitivity = np.divide(
        0.5 * points[..., 2], disparity, out=np.full(disparity.shape, np.nan), where=raw
    )
    return dict(
        left=rect[0],
        right=rect[1],
        points=points,
        disparity=disparity,
        raw_valid=raw,
        valid=valid,
        texture_std=texture,
        roundtrip_error_px=error,
        half_pixel_depth_sensitivity_m=sensitivity,
    )


def polygon_mask(shape, polygons, *, erosion_pixels=0):
    mask = np.zeros(shape, np.uint8)
    if len(shape) != 2 or type(erosion_pixels) is not int or erosion_pixels < 0:
        raise ValueError("Image shape and nonnegative integer erosion required")
    for polygon in polygons:
        p = np.asarray(polygon, float)
        if p.ndim != 2 or p.shape[1] != 2 or len(p) < 3 or not np.isfinite(p).all():
            raise ValueError("Finite image polygons required")
        if (p < 0).any() or (p[:, 0] >= shape[1]).any() or (p[:, 1] >= shape[0]).any():
            raise ValueError("Polygon outside image")
        cv2.fillPoly(mask, [np.rint(p).astype(np.int32)], 1)
    if erosion_pixels:
        mask = cv2.erode(mask, np.ones((2 * erosion_pixels + 1,) * 2, np.uint8))
    return mask.astype(bool)


def propagate_mask(source_gray, target_gray, mask):
    """Image-only local motion proposal; caller must inspect its overlays."""
    if source_gray.shape != target_gray.shape or mask.shape != source_gray.shape:
        raise ValueError("Matching image/mask grids required")
    flow = cv2.calcOpticalFlowFarneback(
        target_gray, source_gray, None, 0.5, 4, 25, 5, 7, 1.5, 0
    )
    y, x = np.indices(mask.shape, dtype=np.float32)
    return cv2.remap(
        mask.astype(np.uint8),
        x + flow[..., 0],
        y + flow[..., 1],
        cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
    ).astype(bool)


def surface_mesh(points, valid, *, maximum_edge_m=0.012):
    """Adjacent observed triangles only; reject holes and depth jumps."""
    import trimesh

    p, v = np.asarray(points, float), np.array(valid, dtype=bool, copy=True)
    if p.shape != (*v.shape, 3) or v.ndim != 2 or not 0 < maximum_edge_m <= 0.05:
        raise ValueError("Organized metric grid and bounded edge length required")
    v &= np.isfinite(p).all(-1)
    ids = np.arange(v.size).reshape(v.shape)
    a, b, c, d = ids[:-1, :-1], ids[:-1, 1:], ids[1:, :-1], ids[1:, 1:]
    triangles = np.concatenate(
        [np.stack([a, c, b], -1).reshape(-1, 3), np.stack([b, c, d], -1).reshape(-1, 3)]
    )
    triangles = triangles[v.ravel()[triangles].all(1)]
    xyz = p.reshape(-1, 3)[triangles]
    edges = np.linalg.norm(xyz - np.roll(xyz, 1, axis=1), axis=2)
    triangles = triangles[(edges.max(1) <= maximum_edge_m) & (edges.min(1) > 1e-8)]
    used, remapped = np.unique(triangles, return_inverse=True)
    return trimesh.Trimesh(
        p.reshape(-1, 3)[used], remapped.reshape(-1, 3), process=False
    )


def voxel_points(points, voxel_m=0.002):
    """Keep actual observed samples; do not average points across surface gaps."""
    p = np.asarray(points, float)
    if (
        p.ndim != 2
        or p.shape[1] != 3
        or not np.isfinite(p).all()
        or not 0 < voxel_m <= 0.02
    ):
        raise ValueError("Finite points and bounded voxel size required")
    _, indices = np.unique(
        np.floor(p / voxel_m).astype(np.int64), axis=0, return_index=True
    )
    return p[np.sort(indices)]


def sample_visibility(
    points_world, world_from_rectified, projection, depth, valid, *, margin_m=0.005
):
    """Classify samples as observed free, occupied/near-surface, or unknown.

    Unknown includes no reliable depth and outside the field of view. It must
    never be treated as clear. Samples behind the first surface are occluded,
    not proof of solid occupancy.
    """
    if not np.isfinite(margin_m) or margin_m <= 0:
        raise ValueError("Positive visibility margin required")
    pose = rigid(world_from_rectified)
    p = transform(points_world, np.linalg.inv(pose))
    projection = np.asarray(projection, float)
    depth, valid = np.asarray(depth), np.asarray(valid, bool)
    if projection.shape != (3, 4) or depth.shape != valid.shape or depth.ndim != 2:
        raise ValueError("Matching depth/validity and 3x4 projection required")
    q = np.c_[p, np.ones(len(p))] @ projection.T
    front = np.isfinite(q).all(1) & (q[:, 2] > 0)
    uv = np.zeros((len(p), 2), dtype=int)
    uv[front] = np.rint(q[front, :2] / q[front, 2:]).astype(int)
    inside = (
        front
        & (uv[:, 0] >= 0)
        & (uv[:, 0] < depth.shape[1])
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < depth.shape[0])
    )
    result = np.zeros(len(p), np.int8)  # 0 unknown; 1 free; 2 surface/occluded
    indices = np.flatnonzero(inside)
    x, y = uv[indices].T
    indices = indices[valid[y, x] & np.isfinite(depth[y, x]) & (depth[y, x] > 0)]
    x, y = uv[indices].T
    result[indices] = np.where(p[indices, 2] < depth[y, x] - margin_m, 1, 2)
    return result


def finger_probe(
    center, normal, *, length_m=0.03, radius_m=0.005, endpoint_gap_m=0.006
):
    """Sample a generic straight finger approach; not an articulated hand."""
    c, n = np.asarray(center, float), np.asarray(normal, float)
    if (
        c.shape != (3,)
        or n.shape != (3,)
        or not np.isfinite([c, n]).all()
        or not np.isclose(np.linalg.norm(n), 1)
        or not 0 < radius_m < length_m <= 0.15
        or not 0 < endpoint_gap_m < length_m
    ):
        raise ValueError(
            "Finite center, unit normal and bounded probe dimensions required"
        )
    axis = np.eye(3)[np.argmin(abs(n))]
    a = np.cross(n, axis)
    a /= np.linalg.norm(a)
    b = np.cross(n, a)
    theta = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    cross = np.r_[
        np.zeros((1, 3)),
        radius_m * (np.cos(theta)[:, None] * a + np.sin(theta)[:, None] * b),
    ]
    along = np.linspace(endpoint_gap_m, length_m, 9)
    return (c + along[:, None, None] * n + cross[None]).reshape(-1, 3)
