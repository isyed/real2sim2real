"""One shared metric camera-depth contract for mesh generation and tracking."""

import numpy as np


def pointmap(depth_m, intrinsics):
    z = np.asarray(depth_m, dtype=np.float32)
    K = np.asarray(intrinsics, dtype=np.float64)
    if (
        z.ndim != 2
        or K.shape != (3, 3)
        or not np.isfinite(K).all()
        or min(K[0, 0], K[1, 1]) <= 0
        or not np.allclose(K[2], [0, 0, 1])
    ):
        raise ValueError(
            "Native camera-Z metres and a pixel intrinsics matrix are required"
        )
    y, x = np.indices(z.shape)
    rays = np.stack(
        ((x - K[0, 2]) / K[0, 0], (y - K[1, 2]) / K[1, 1], np.ones_like(z)), axis=-1
    )
    valid = np.isfinite(z) & (z > 0) & (z < 65.535)
    xyz = (rays * z[..., None]).astype(np.float32)
    xyz[~valid] = np.nan
    return xyz


def check_pointmap(xyz, intrinsics, mask):
    xyz = np.asarray(xyz)
    m = np.asarray(mask, dtype=bool)
    K = np.asarray(intrinsics)
    if xyz.shape != (*m.shape, 3) or K.shape != (3, 3):
        raise ValueError("RGB/mask/XYZ must use the native grid")
    valid = np.isfinite(xyz).all(-1) & (xyz[..., 2] > 0)
    if (valid & m).sum() < 100:
        raise ValueError("Insufficient foreground metric depth")
    y, x = np.indices(m.shape)
    z = xyz[..., 2][valid]
    error = max(
        abs(xyz[..., 0][valid] * K[0, 0] / z + K[0, 2] - x[valid]).max(),
        abs(xyz[..., 1][valid] * K[1, 1] / z + K[1, 2] - y[valid]).max(),
    )
    if error > 0.001:
        raise ValueError("Metric point-map rays disagree with the camera")
    return {
        "valid_object_pixels": int((valid & m).sum()),
        "maximum_projection_error_px": float(error),
    }
