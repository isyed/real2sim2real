"""CPU triangle z-buffer for auditing predicted object depth in metres.

Integer image coordinates are pixel centres, matching calibrated camera K.
Unlike sparse vertex projection, this measures the first triangle surface on
each ray. It does not validate monocular depth or fill missing observations.
"""
import numpy as np


def compare_surface_depth(surface_m, predicted_m, object_mask, *, occluder_mask=None,
                          erosion_pixels=0, min_points=100, max_error_m=.10,
                          depth_range_m=(.001, 8.)):
    """Compare visible first-surface Z with predicted depth, failing closed.

    This is a consistency test against a prediction, never metric validation or
    candidate approval. Erosion excludes uncertain segmentation edges and is
    explicit in the receipt. Missing/occluded pixels cannot count as agreement.
    """
    from scipy.ndimage import binary_erosion
    surface, predicted = np.asarray(surface_m, float), np.asarray(predicted_m, float)
    mask = np.asarray(object_mask)
    if surface.ndim != 2 or surface.shape != predicted.shape or mask.shape != surface.shape or mask.dtype != bool:
        raise ValueError('Expected matching depth grids and a boolean object mask')
    if occluder_mask is not None:
        occ = np.asarray(occluder_mask)
        if occ.shape != mask.shape or occ.dtype != bool:
            raise ValueError('Expected matching boolean occluder mask')
        mask = mask & ~occ
    if type(erosion_pixels) is not int or erosion_pixels < 0 or type(min_points) is not int or min_points < 1:
        raise ValueError('Invalid erosion or evidence count')
    if not np.isfinite(max_error_m) or max_error_m <= 0:
        raise ValueError('Invalid metric error limit')
    limits = np.asarray(depth_range_m, float)
    if limits.shape != (2,) or not np.isfinite(limits).all() or not 0 <= limits[0] < limits[1]:
        raise ValueError('Invalid metric depth interval')
    if erosion_pixels:
        mask = binary_erosion(mask, iterations=erosion_pixels)
    valid = mask & np.isfinite(surface) & (surface > 0) & np.isfinite(predicted) & (predicted > limits[0]) & (predicted < limits[1])
    residual = predicted[valid]-surface[valid]
    median = float(np.median(np.abs(residual))) if len(residual) else None
    enough = len(residual) >= min_points
    return dict(points=int(len(residual)), median_absolute_m=median,
                median_signed_m=float(np.median(residual)) if len(residual) else None,
                p90_absolute_m=float(np.quantile(np.abs(residual), .9)) if len(residual) else None,
                sufficient_evidence=bool(enough),
                consistency_passed=bool(enough and median <= max_error_m),
                erosion_pixels=erosion_pixels, min_points=min_points, max_error_m=float(max_error_m),
                metric_truth_validated=False, candidate_approved=False)


def front_surface_depth(vertices_camera, faces, intrinsics, image_shape, *, batch_samples=262144):
    """Return camera-Z metres; unrendered pixels are infinity.

    Perspective-correct interpolation, both triangle windings, no backface
    culling. Near-plane crossings are rejected rather than silently clipped.
    Memory for raster samples is bounded independently of triangle count.
    """
    v = np.asarray(vertices_camera, dtype=float)
    f = np.asarray(faces)
    k = np.asarray(intrinsics, dtype=float)
    if v.ndim != 2 or v.shape[1] != 3 or not len(v) or not np.isfinite(v).all():
        raise ValueError('Expected finite camera-space vertices')
    if f.ndim != 2 or f.shape[1] != 3 or f.dtype.kind not in 'iu' or (f.size and (f.min() < 0 or f.max() >= len(v))):
        raise ValueError('Invalid triangle indices')
    if k.shape != (3, 3) or not np.isfinite(k).all() or not np.allclose(k[2], [0, 0, 1]) or np.linalg.det(k) <= 0:
        raise ValueError('Expected calibrated pinhole intrinsics')
    if len(image_shape) != 2 or any(type(n) is not int or n <= 0 for n in image_shape):
        raise ValueError('Expected positive integer image dimensions')
    if type(batch_samples) is not int or batch_samples <= 0:
        raise ValueError('Expected positive sample batch size')
    if (v[:, 2] <= 1e-5).any():
        raise ValueError('Mesh crosses camera near plane')
    h, w = image_shape
    out = np.full(h*w, np.inf)
    p = v @ k.T
    uv = p[:, :2] / p[:, 2:]
    triangles = uv[f]
    lo = np.maximum(np.ceil(triangles.min(1)), [0, 0]).astype(np.int64)
    hi = np.minimum(np.floor(triangles.max(1)), [w-1, h-1]).astype(np.int64)
    a, b, c = triangles.transpose(1, 0, 2)
    denominator = (b[:, 1]-c[:, 1])*(a[:, 0]-c[:, 0]) + (c[:, 0]-b[:, 0])*(a[:, 1]-c[:, 1])
    keep = (hi >= lo).all(1) & (np.abs(denominator) > 1e-12)
    lo, hi, a, b, c, denominator, f = [x[keep] for x in (lo, hi, a, b, c, denominator, f)]
    width = hi[:, 0]-lo[:, 0]+1
    count = width*(hi[:, 1]-lo[:, 1]+1)
    offsets = np.r_[0, np.cumsum(count)]
    inverse_z = 1/v[f, 2]
    for start in range(0, int(offsets[-1]), batch_samples):
        sample = np.arange(start, min(start+batch_samples, int(offsets[-1])))
        t = np.searchsorted(offsets[1:], sample, side='right')
        local = sample-offsets[t]
        x = lo[t, 0]+local % width[t]
        y = lo[t, 1]+local // width[t]
        l0 = ((b[t, 1]-c[t, 1])*(x-c[t, 0])+(c[t, 0]-b[t, 0])*(y-c[t, 1]))/denominator[t]
        l1 = ((c[t, 1]-a[t, 1])*(x-c[t, 0])+(a[t, 0]-c[t, 0])*(y-c[t, 1]))/denominator[t]
        l2 = 1-l0-l1
        inside = (l0 >= -1e-10) & (l1 >= -1e-10) & (l2 >= -1e-10)
        iz = l0*inverse_z[t, 0]+l1*inverse_z[t, 1]+l2*inverse_z[t, 2]
        np.minimum.at(out, (y*w+x)[inside], 1/iz[inside])
    return out.reshape(h, w)
