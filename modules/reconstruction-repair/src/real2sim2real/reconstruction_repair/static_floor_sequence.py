"""Audit a static view and fit bounded depth gains without hiding failed frames.

Numeric callbacks keep video/storage conventions in the caller. This produces
depth evidence only: mesh poses, human motion and metric accuracy are not
certified. A corrected depth sequence must be followed by fresh pose validation.
"""
import numpy as np

from .depth_scale import static_floor_gain
from .metric_floor import floor_points


def inspect_static_patches(reference, current, rectangles, *, min_correlation=.90,
                           max_displacement_pixels=2, search_radius_pixels=6):
    """Check reviewed stationary image regions with a bounded translation search.

    This is sampled evidence for a static view, not a full camera-motion model.
    Low texture, insufficient coverage and any disagreeing region fail closed.
    """
    a, b = np.asarray(reference, float), np.asarray(current, float)
    if (a.ndim != 2 or a.shape != b.shape or not np.isfinite(a).all()
            or not np.isfinite(b).all() or len(rectangles) < 2
            or not 0 < min_correlation <= 1
            or not 0 <= max_displacement_pixels < search_radius_pixels
            or int(search_radius_pixels) != search_radius_pixels):
        raise ValueError('Finite native grayscale grids and bounded patch search required')
    radius = int(search_radius_pixels)
    rows = []
    for bounds in rectangles:
        if len(bounds) != 4 or any(int(x) != x for x in bounds):
            raise ValueError('Integer XYXY patch bounds required')
        x0, y0, x1, y1 = map(int, bounds)
        if not (0 <= x0 < x1 <= a.shape[1] and 0 <= y0 < y1 <= a.shape[0]
                and min(x1-x0, y1-y0) > 2*radius+8):
            raise ValueError('Patch outside native image or too small for search')
        anchor = a[y0+radius:y1-radius:2, x0+radius:x1-radius:2]
        anchor = anchor-anchor.mean()
        norm = float(np.linalg.norm(anchor))
        best = (-float('inf'), 0, 0)
        for dy in range(-radius, radius+1):
            for dx in range(-radius, radius+1):
                patch = b[y0+radius+dy:y1-radius+dy:2, x0+radius+dx:x1-radius+dx:2]
                patch = patch-patch.mean()
                denominator = norm*float(np.linalg.norm(patch))
                score = float(np.sum(anchor*patch)/denominator) if denominator > 1e-9 else -1.
                if score > best[0]:
                    best = (score, dx, dy)
        texture = float(np.std(anchor))
        displacement = float(np.hypot(best[1], best[2]))
        rows.append(dict(correlation=best[0], dx_pixels=best[1], dy_pixels=best[2],
                         texture_std=texture, passed=bool(texture >= 2 and
                         best[0] >= min_correlation and displacement <= max_displacement_pixels)))
    return dict(passed=all(r['passed'] for r in rows), regions=rows,
                scope='Reviewed static image patches only; excludes unmodelled rotation/parallax',
                thresholds=dict(min_correlation=min_correlation,
                                max_displacement_pixels=max_displacement_pixels,
                                search_radius_pixels=search_radius_pixels))


def fit_sequence(frame_count, read_frame, floor_selection, anchor_plane, first_gain,
                 *, camera_evidence, pixel_stride=8):
    """Fit each raw depth frame to one original predicted floor anchor.

    ``read_frame(i)`` returns raw camera-Z metres, K, human mask, object mask.
    Rejected frames remain rejected; no gain interpolation or last-good filling.
    The exact original frame-zero gain is retained. All original gain/region/
    residual limits in ``static_floor_gain`` remain unchanged.
    """
    if (int(frame_count) != frame_count or frame_count < 2 or
            int(pixel_stride) != pixel_stride or pixel_stride < 1 or
            camera_evidence.get('passed') is not True or not camera_evidence.get('source_sha256')):
        raise ValueError('Complete timeline, positive stride and hashed static-camera evidence required')
    rows = []
    for index in range(int(frame_count)):
        try:
            raw, K, human, obj = read_frame(index)
            points, regions, _ = floor_points(raw, K, human, obj, floor_selection)
            # floor_points emits each rectangle in raster order. Subsampling
            # within each region keeps both independent ROIs represented.
            ids = np.concatenate([np.flatnonzero(regions == region)[::pixel_stride**2]
                                  for region in np.unique(regions)])
            gain, diagnostics = static_floor_gain(points[ids], regions[ids], anchor_plane,
                                                   first_gain=first_gain)
            if index == 0:
                # Check the exact retained anchor, not just the fitted gain.
                residual = np.abs(points[ids] @ np.asarray(anchor_plane)[:3]*first_gain
                                  + np.asarray(anchor_plane)[3])
                if np.percentile(residual, 95) > .02:
                    raise ValueError('Original frame-zero gain fails its floor anchor')
                gain = float(first_gain)
            rows.append(dict(frame_index=index, passed=True, gain=gain, **diagnostics))
        except ValueError as error:
            rows.append(dict(frame_index=index, passed=False, gain=None, error=str(error)))
    failed = [r['frame_index'] for r in rows if not r['passed']]
    return dict(schema='real2sim2real.static-floor-sequence/1', complete=not failed,
                frames=rows, failed_frame_indices=failed, first_gain=float(first_gain),
                camera_evidence=camera_evidence, raw_depth_required=True,
                original_frame_zero_gain_preserved=True, failed_frames_filled=False,
                pixel_stride=pixel_stride, approved_for_use=False,
                requires_pose_revalidation=True, human_changed=False,
                absolute_metric_accuracy_established=False)
