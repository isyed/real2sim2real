"""Image/depth evidence for declared object-local contact points, without fitting."""

import numpy as np

from .camera import rigid, transform


def observe_contacts(
    points_object,
    normals_object,
    camera_from_object,
    intrinsics,
    depth,
    object_mask,
    occluder_mask,
    *,
    radius_pixels=2,
    minimum_depth_pixels=5,
    minimum_facing_cosine=0.25,
):
    """Keep absent/occluded evidence separate from depth disagreement.

    Visibility here uses normals and image masks; a caller must additionally
    check mesh self-occlusion. Depth is optical-axis distance in metres.
    """
    points, normals = (
        np.asarray(points_object, float),
        np.asarray(normals_object, float),
    )
    k = np.asarray(intrinsics, float)
    depth, mask, occluder = (
        np.asarray(depth),
        np.asarray(object_mask),
        np.asarray(occluder_mask),
    )
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or normals.shape != points.shape
        or not np.isfinite(points).all()
        or not np.isfinite(normals).all()
        or not np.allclose(np.linalg.norm(normals, axis=1), 1)
        or k.shape != (3, 3)
        or not np.isfinite(k).all()
        or min(k[0, 0], k[1, 1]) <= 0
        or depth.ndim != 2
        or mask.shape != depth.shape
        or occluder.shape != depth.shape
        or not isinstance(radius_pixels, int)
        or not 0 <= radius_pixels <= 5
        or not 1 <= minimum_depth_pixels <= (2 * radius_pixels + 1) ** 2
        or not 0 <= minimum_facing_cosine < 1
    ):
        raise ValueError(
            "Finite calibrated contacts and bounded image sampling required"
        )
    rigid(np.asarray(camera_from_object))
    camera = transform(points, camera_from_object)
    camera_normals = normals @ np.asarray(camera_from_object)[:3, :3].T
    rows = []
    height, width = depth.shape
    for p, n in zip(camera, camera_normals):
        row = dict(
            observed=False,
            pixel=None,
            expected_depth_m=float(p[2]),
            depth_pixels=0,
            median_depth_m=None,
            residual_m=None,
            depth_mad_m=None,
        )
        if p[2] <= 0:
            row["reason"] = "behind_camera"
            rows.append(row)
            continue
        uv = (k @ p)[:2] / p[2]
        row["pixel"] = uv.tolist()
        x, y = np.rint(uv).astype(int)
        facing = float(n @ -p / np.linalg.norm(p))
        row["facing_cosine"] = facing
        if not (0 <= x < width and 0 <= y < height):
            row["reason"] = "outside_image"
        elif facing < minimum_facing_cosine:
            row["reason"] = "backfacing_or_grazing"
        elif occluder[y, x]:
            row["reason"] = "masked_occluder"
        elif not mask[y, x]:
            row["reason"] = "outside_object_mask"
        else:
            r = radius_pixels
            window = np.s_[
                max(0, y - r) : min(height, y + r + 1),
                max(0, x - r) : min(width, x + r + 1),
            ]
            z = depth[window]
            valid = (
                mask[window].astype(bool)
                & ~occluder[window].astype(bool)
                & np.isfinite(z)
                & (z > 0)
            )
            values = z[valid]
            row["depth_pixels"] = len(values)
            if len(values) < minimum_depth_pixels:
                row["reason"] = "insufficient_depth"
            else:
                median = float(np.median(values))
                row.update(
                    observed=True,
                    reason="observed",
                    median_depth_m=median,
                    residual_m=median - float(p[2]),
                    depth_mad_m=float(np.median(abs(values - median))),
                )
        rows.append(row)
    return rows


def summarize_contact(
    rows, *, minimum_frames=3, max_median_error_m=0.005, max_p90_error_m=0.01
):
    if (
        not isinstance(minimum_frames, int)
        or minimum_frames < 1
        or not np.isfinite([max_median_error_m, max_p90_error_m]).all()
        or not 0 < max_median_error_m <= max_p90_error_m
    ):
        raise ValueError("Explicit finite observation criteria required")
    values = np.array([r["residual_m"] for r in rows if r["observed"]], float)
    if not np.isfinite(values).all():
        raise ValueError("Observed depth evidence must be finite")
    enough = len(values) >= minimum_frames
    median = float(np.median(abs(values))) if len(values) else None
    p90 = float(np.quantile(abs(values), 0.9)) if len(values) else None
    return dict(
        observed_frames=len(values),
        minimum_frames=minimum_frames,
        median_abs_depth_error_m=median,
        p90_abs_depth_error_m=p90,
        evidence="supported"
        if enough and median <= max_median_error_m and p90 <= max_p90_error_m
        else "inconsistent"
        if enough
        else "insufficient",
        independent_ground_truth=False,
    )
