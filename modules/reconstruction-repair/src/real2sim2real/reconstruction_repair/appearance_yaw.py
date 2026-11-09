"""CPU RGB-cued yaw proposals for a fixed coloured mesh.

This is a proposal generator, not a tracker or acceptance gate. Inputs must be
source-derived. Native triangle/depth and independent-frame validation remain
the caller's responsibility. No mesh size, appearance or timestamp is fitted.
"""
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import map_coordinates
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation


def yaw_pose(rotation, translation, centroid, up_camera, angle_deg):
    """Rotate about the existing camera-space mesh centroid, preserving it."""
    r, t, c, up = map(lambda a: np.asarray(a, float), (rotation, translation, centroid, up_camera))
    if r.shape != (3, 3) or t.shape != (3,) or c.shape != (3,) or up.shape != (3,):
        raise ValueError("Explicit rigid pose, local centroid and camera up required")
    if not all(np.isfinite(x).all() for x in (r, t, c, up)) or not np.isfinite(angle_deg):
        raise ValueError("Finite yaw proposal inputs required")
    if not np.isclose(np.linalg.norm(up), 1., atol=1e-6):
        raise ValueError("Unit camera-space up axis required")
    if not np.allclose(r @ r.T, np.eye(3), atol=1e-4) or not np.isclose(np.linalg.det(r), 1., atol=1e-4):
        raise ValueError("Proper source rotation required")
    candidate = Rotation.from_rotvec(up * np.deg2rad(angle_deg)).as_matrix() @ r
    return candidate, t + r @ c - candidate @ c


def image_cost(vertices, colors, rotation, translation, intrinsics, rgb, object_mask, human_mask, *, min_pixels=300):
    """Approximate visible-vertex RGB loss; proposal-resolution evidence only.

    The caller explicitly selects a fixed vertex sample. One nearest surface
    sample per rounded native pixel prevents dense/back vertices dominating.
    A bounded affine exposure/linear illumination fit tolerates lighting changes.
    No fit is usable without sufficient nonhuman foreground observations.
    """
    v, colors, k = np.asarray(vertices, float), np.asarray(colors, float), np.asarray(intrinsics, float)
    rgb = np.asarray(rgb, float)
    mask, human = np.asarray(object_mask), np.asarray(human_mask)
    if (v.ndim != 2 or v.shape[1] != 3 or colors.shape != v.shape or
            rgb.ndim != 3 or rgb.shape[2] != 3 or mask.shape != rgb.shape[:2] or human.shape != mask.shape
            or mask.dtype != bool or human.dtype != bool or k.shape != (3, 3)):
        raise ValueError("Matching coloured metric points, native RGB/camera and boolean masks required")
    if not all(np.isfinite(x).all() for x in (v, colors, rgb, k)):
        raise ValueError("Finite source image and model appearance required")
    if colors.min() < 0 or colors.max() > 1 or rgb.min() < 0 or rgb.max() > 1:
        raise ValueError("Explicit RGB intensity range [0,1] required")
    p = v @ np.asarray(rotation).T + np.asarray(translation)
    if np.any(p[:, 2] <= 1e-5):
        raise ValueError("Model crosses camera near plane")
    project = p @ k.T
    uv = project[:, :2] / project[:, 2:]
    xy = np.rint(uv).astype(int)
    h, w = mask.shape
    inside = (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
    ids = np.flatnonzero(inside)
    flat = xy[ids, 1] * w + xy[ids, 0]
    order = np.lexsort((p[ids, 2], flat))
    _, first = np.unique(flat[order], return_index=True)
    visible = ids[order[first]]
    use = mask[xy[visible, 1], xy[visible, 0]] & ~human[xy[visible, 1], xy[visible, 0]]
    visible = visible[use]
    if len(visible) < min_pixels:
        return dict(usable=False, pixels=int(len(visible)), cost=None, reason="insufficient_source_pixels")
    points = uv[visible]
    observed = np.stack([map_coordinates(rgb[:, :, i], [points[:, 1], points[:, 0]], order=1, mode="nearest") for i in range(3)], axis=1)
    gray = np.array([.299, .587, .114])
    model_gray, observed_gray = colors[visible] @ gray, observed @ gray
    centered = (points-points.mean(0)) / max(10., float(np.ptp(points, axis=0).max()))
    design = np.column_stack((model_gray, np.ones(len(visible)), centered))
    weights = 1 + np.minimum(2., np.abs(model_gray-np.median(model_gray))/.06)
    coefficients = np.array([1., 0., 0., 0.])
    for _ in range(3):
        root = np.sqrt(weights)
        coefficients = np.linalg.lstsq(design*root[:, None], observed_gray*root, rcond=None)[0]
        coefficients[0] = np.clip(coefficients[0], .2, 3.)
        coefficients[2:] = np.clip(coefficients[2:], -.4, .4)
        coefficients[1] = np.median(observed_gray-design[:, [0, 2, 3]] @ coefficients[[0, 2, 3]])
        error = observed_gray-design@coefficients
        weights = (1 + np.minimum(2., np.abs(model_gray-np.median(model_gray))/.06)) * np.minimum(1., .08/np.maximum(np.abs(error), 1e-6))
    residual = np.abs(observed_gray-design@coefficients)
    texture_error = float(np.mean(np.minimum(residual, .20) * (1+np.minimum(2., np.abs(model_gray-np.median(model_gray))/.06))))
    # Explicit convex-outline proxy for fitting only; full triangle validation is separate.
    hull = ConvexHull(uv).vertices
    silhouette = Image.new("1", (w, h))
    ImageDraw.Draw(silhouette).polygon([tuple(x) for x in uv[hull]], fill=1)
    silhouette = np.asarray(silhouette, bool) & ~human
    target = mask & ~human
    iou = float(np.count_nonzero(silhouette & target)/max(1, np.count_nonzero(silhouette | target)))
    return dict(usable=True, pixels=int(len(visible)), cost=texture_error+.15*(1-iou),
                photometric_error=texture_error, convex_outline_iou=iou,
                exposure_gain=float(coefficients[0]), exposure_offset=float(coefficients[1]),
                candidate_approved=False, approximate_vertex_visibility=True)


def select_path(costs, angles_deg, frame_ids, *, max_speed_deg_per_frame=6., transition_weight=.003):
    """Bounded circular-state DP; no direction or full-turn count is imposed."""
    costs, angles, frames = np.asarray(costs, float), np.asarray(angles_deg, float), np.asarray(frame_ids)
    if costs.shape != (len(frames), len(angles)) or len(frames) < 2 or len(angles) < 3:
        raise ValueError("Frame/state cost grid required")
    if (np.any(np.diff(frames) <= 0) or not np.isfinite(costs).all() or
            not np.isfinite(angles).all() or not np.isfinite(max_speed_deg_per_frame) or
            max_speed_deg_per_frame <= 0 or not np.isfinite(transition_weight) or transition_weight < 0):
        raise ValueError("Finite observation costs and increasing source frames required")
    delta = (angles[None, :]-angles[:, None]+180) % 360-180
    accumulated = costs[0].copy()
    back = np.zeros(costs.shape, int)
    for i in range(1, len(frames)):
        limit = max_speed_deg_per_frame*(frames[i]-frames[i-1])
        transition = np.where(np.abs(delta) <= limit, transition_weight*(delta/15.)**2, np.inf)
        all_costs = accumulated[:, None]+transition
        back[i] = np.argmin(all_costs, axis=0)
        accumulated = costs[i]+np.min(all_costs, axis=0)
    if not np.isfinite(accumulated).any():
        raise ValueError("No bounded observed yaw path")
    states = np.zeros(len(frames), int)
    states[-1] = np.argmin(accumulated)
    for i in range(len(frames)-1, 0, -1):
        states[i-1] = back[i, states[i]]
    yaw = np.rad2deg(np.unwrap(np.deg2rad(angles[states])))
    return dict(state_indices=states, yaw_deg=yaw, maximum_fit_step_deg=float(np.abs(np.diff(yaw)).max()),
                candidate_approved=False, full_reference_accepted=False)


def landmark_cost(points_local, outward_normals_local, observed_pixels, rotation,
                  translation, intrinsics, *, pixel_sigma=(2., 2.)):
    """Explicit source-reviewed surface landmarks with a front-facing check.

    This checks reprojection and local facing, not full-mesh self-occlusion.
    A caller must establish semantic correspondence and independently check
    native silhouettes/depth. Hidden landmarks must not be invented as labels.
    """
    points, normals, observed, r, t, k, sigma = map(
        lambda x: np.asarray(x, float),
        (points_local, outward_normals_local, observed_pixels, rotation,
         translation, intrinsics, pixel_sigma))
    if (points.ndim != 2 or points.shape[1] != 3 or normals.shape != points.shape
            or observed.shape != (len(points), 2) or r.shape != (3, 3) or t.shape != (3,)
            or k.shape != (3, 3) or sigma.shape != (2,) or (sigma <= 0).any()):
        raise ValueError("Corresponding metric landmarks, source pixels and explicit pixel uncertainty required")
    if not all(np.isfinite(x).all() for x in (points, normals, observed, r, t, k, sigma)):
        raise ValueError("Finite landmark observations required")
    if not np.allclose(np.linalg.norm(normals, axis=1), 1., atol=1e-4):
        raise ValueError("Unit outward surface normals required")
    camera = points @ r.T+t
    front = (camera[:, 2] > 1e-5) & (np.einsum("ij,ij->i", normals @ r.T, camera) < 0)
    if not front.all():
        return dict(usable=False, cost=None, reason="landmark_not_front_facing",
                    candidate_approved=False)
    projected = camera @ k.T
    projected = projected[:, :2]/projected[:, 2:]
    residual = projected-observed
    return dict(usable=True, cost=float(np.mean(np.sum((residual/sigma)**2, axis=1))),
                projected_pixels=projected.tolist(), residual_pixels=residual.tolist(),
                pixel_rms=float(np.sqrt(np.mean(np.sum(residual**2, axis=1)))),
                horizontal_rms=float(np.sqrt(np.mean(residual[:, 0]**2))),
                candidate_approved=False, full_self_occlusion_checked=False)


def centroid_translation(vertices, rotation, translation, intrinsics, object_mask,
                         human_mask, up_camera, *, max_move_m=.08, iterations=2,
                         plane_semantics="gravity"):
    """Bounded proposal-plane translation from a convex silhouette centroid.

    Motion is restricted to the plane perpendicular to the supplied normal.
    Explicit gravity mode preserves resting height; camera-depth mode requires
    the optical axis and preserves Z instead, with no height claim. This proxy
    cannot authorize a candidate or replace native triangle/depth validation.
    """
    v, r, initial, k, up = map(lambda x: np.asarray(x, float),
                               (vertices, rotation, translation, intrinsics, up_camera))
    mask, human = np.asarray(object_mask), np.asarray(human_mask)
    if (v.ndim != 2 or v.shape[1] != 3 or r.shape != (3, 3) or initial.shape != (3,)
            or k.shape != (3, 3) or mask.ndim != 2 or human.shape != mask.shape
            or mask.dtype != bool or human.dtype != bool or up.shape != (3,)
            or not np.isclose(np.linalg.norm(up), 1., atol=1e-6)
            or not np.isfinite(max_move_m) or max_move_m <= 0
            or type(iterations) is not int or iterations < 1):
        raise ValueError("Explicit metric geometry, native masks, unit gravity and bounded iteration count required")
    if not all(np.isfinite(x).all() for x in (v, r, initial, k, up)):
        raise ValueError("Finite centroid proposal inputs required")
    if plane_semantics not in ("gravity", "camera_depth"):
        raise ValueError("Explicit plane semantics required")
    if plane_semantics == "camera_depth" and not np.array_equal(up, [0., 0., 1.]):
        raise ValueError("Camera-depth plane requires the optical axis")
    target = mask & ~human
    yy, xx = np.nonzero(target)
    if len(xx) < 100:
        return dict(usable=False, reason="insufficient_source_pixels", candidate_approved=False)
    observed = np.array([xx.mean(), yy.mean()])
    t = initial.copy()
    for _ in range(iterations):
        p = v@r.T+t
        if np.any(p[:, 2] <= 1e-5):
            return dict(usable=False, reason="camera_plane", candidate_approved=False)
        q = p@k.T
        uv = q[:, :2]/q[:, 2:]
        hull = ConvexHull(uv).vertices
        raster = Image.new("1", (mask.shape[1], mask.shape[0]))
        ImageDraw.Draw(raster).polygon([tuple(x) for x in uv[hull]], fill=1)
        visible = np.asarray(raster, bool) & ~human
        sy, sx = np.nonzero(visible)
        if len(sx) < 100:
            return dict(usable=False, reason="insufficient_projected_pixels", candidate_approved=False)
        displacement = observed-np.array([sx.mean(), sy.mean()])
        centre = p.mean(0)
        projected = k@centre
        jacobian = (k[:2]*projected[2]-projected[:2, None]*k[2])/projected[2]**2
        system = np.vstack((jacobian, up))
        if np.linalg.cond(system) > 1e6:
            return dict(usable=False, reason="plane_projection_degenerate", candidate_approved=False)
        t += np.linalg.solve(system, np.r_[displacement, 0.])
        if np.linalg.norm(t-initial) > max_move_m:
            return dict(usable=False, reason="translation_bound", candidate_approved=False)
    return dict(usable=True, translation=t.tolist(), move_m=float(np.linalg.norm(t-initial)),
                height_change_m=float((t-initial)@up) if plane_semantics == "gravity" else None,
                constrained_plane=plane_semantics, plane_coordinate_change_m=float((t-initial)@up),
                candidate_approved=False,
                convex_proxy=True, native_validation_required=True)


def project_vertex_colors(vertices, colors, rotation, translation, intrinsics, image_shape):
    """Dense nearest-front-vertex preview, not a triangle acceptance renderer."""
    v, c, k = np.asarray(vertices, float), np.asarray(colors), np.asarray(intrinsics, float)
    if v.ndim != 2 or v.shape[1] != 3 or c.shape != v.shape or c.dtype != np.uint8:
        raise ValueError("Metric vertices and matching uint8 RGB required")
    if len(image_shape) != 2 or any(type(x) is not int or x < 1 for x in image_shape):
        raise ValueError("Positive native image shape required")
    h, w = image_shape
    p = v@np.asarray(rotation).T+np.asarray(translation)
    if not np.isfinite(p).all() or np.any(p[:, 2] <= 1e-5):
        raise ValueError("Finite mesh wholly in front of camera required")
    q = p@k.T
    xy = np.rint(q[:, :2]/q[:, 2:]).astype(int)
    valid = (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
    ids = np.flatnonzero(valid)
    flat = xy[ids, 1]*w+xy[ids, 0]
    order = np.lexsort((p[ids, 2], flat))
    _, first = np.unique(flat[order], return_index=True)
    ids = ids[order[first]]
    rgb = np.zeros((h, w, 3), np.uint8)
    rgb[xy[ids, 1], xy[ids, 0]] = c[ids]
    return rgb
