"""Explicit OpenCV camera geometry, in metres and column-vector convention."""

from __future__ import annotations

import numpy as np


def rigid(value):
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("Expected a finite 4x4 rigid transform")
    if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-7):
        raise ValueError("Invalid homogeneous transform")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-5) or not np.isclose(
        np.linalg.det(rotation), 1, atol=2e-5
    ):
        raise ValueError("Expected a proper rotation without scale")
    return matrix


def transform(points, target_from_source):
    t = rigid(target_from_source)
    return np.asarray(points) @ t[:3, :3].T + t[:3, 3]


def intrinsic(value):
    k = np.asarray(value, dtype=np.float64)
    if (
        k.shape != (3, 3)
        or not np.isfinite(k).all()
        or min(k[0, 0], k[1, 1]) <= 0
        or not np.allclose(k[2], [0, 0, 1])
    ):
        raise ValueError("Expected finite positive pinhole intrinsics")
    return k


def resize_intrinsic(k, source_size, output_size):
    """Pixel-centre preserving resize, matching OpenCV's resize convention."""
    k = intrinsic(k).copy()
    sx, sy = np.asarray(output_size) / np.asarray(source_size)
    k[0] *= sx
    k[1] *= sy
    k[0, 2] += (sx - 1) / 2
    k[1, 2] += (sy - 1) / 2
    return k


def unproject(depth, k):
    k = intrinsic(k)
    y, x = np.indices(depth.shape)
    rays = np.stack([x, y, np.ones_like(x)], -1) @ np.linalg.inv(k).T
    return rays * np.asarray(depth)[..., None]


def project_depth(points, k, size):
    """Z-buffer projection; unseen pixels remain zero, never fabricated depth."""
    k = intrinsic(k)
    width, height = map(int, size)
    p = np.asarray(points).reshape(-1, 3)
    p = p[np.isfinite(p).all(1) & (p[:, 2] > 0)]
    uv = p @ k.T
    uv = np.rint(uv[:, :2] / uv[:, 2:]).astype(np.int64)
    valid = (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    flat = np.full(width * height, np.inf, dtype=np.float32)
    np.minimum.at(flat, uv[valid, 1] * width + uv[valid, 0], p[valid, 2])
    flat[~np.isfinite(flat)] = 0
    return flat.reshape(height, width)
