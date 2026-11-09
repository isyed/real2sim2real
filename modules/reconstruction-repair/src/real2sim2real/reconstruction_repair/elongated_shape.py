"""Flag likely missing crossbars in slender object meshes before pose tracking.

This is a source-mask/depth diagnostic, not a mesh repair. It compares a robust
local width of a visible object mask with the mesh's second principal extent.
Pose, foreshortening and predicted depth can affect the comparison, so a flag
requires visual review rather than automatic mesh replacement.
"""

import numpy as np


def principal_axis(vertices):
    """Retain one mesh-local axis for bounded-memory native-frame diagnostics."""
    vertices=np.asarray(vertices,float)
    if vertices.ndim!=2 or vertices.shape[1]!=3 or len(vertices)<4 or not np.isfinite(vertices).all():
        raise ValueError('Finite metric mesh vertices required')
    center=vertices.mean(0);_,values,directions=np.linalg.svd(vertices-center,full_matrices=False)
    if values[0]<=0:raise ValueError('Nonzero mesh extent required')
    axis=directions[0];projection=(vertices-center)@axis
    return dict(center_mesh=center.tolist(),axis_mesh=axis.tolist(),
        half_length_m=float(np.ptp(projection)/2),mesh_axis_ratio=float(values[0]/max(values[1],1e-12)))


def orientation_audit(axis,pose_camera,intrinsics,object_mask,human_mask,*,min_pixels=100,min_axis_ratio=2.):
    """Compare a visible mask's unoriented axis with its proposed projection.

    A line direction does not identify head/tail or roll about the mesh axis.
    Near-end-on or heavily occluded views are explicitly uninformative. This
    function never changes poses, masks, depth or native acceptance gates.
    """
    from real2sim2real.reconstruction.recovery_quality import rigid
    pose=np.asarray(pose_camera,float);k=np.asarray(intrinsics,float)
    obj,human=np.asarray(object_mask,bool),np.asarray(human_mask,bool)
    center=np.asarray(axis['center_mesh'],float);direction=np.asarray(axis['axis_mesh'],float);length=float(axis['half_length_m'])
    if (not rigid(pose) or k.shape!=(3,3) or not np.isfinite(k).all() or min(k[0,0],k[1,1])<=0
            or obj.ndim!=2 or human.shape!=obj.shape or min_pixels<3 or min_axis_ratio<=1
            or center.shape!=(3,) or direction.shape!=(3,) or not np.isfinite([*center,*direction,length]).all()
            or length<=0 or not np.isclose(np.linalg.norm(direction),1.)):
        raise ValueError('Rigid camera pose, calibrated native masks and retained mesh axis required')
    base=dict(schema='real2sim2real.elongated-orientation-audit/1',diagnostic_only=True,
        head_tail_distinguished=False,axial_roll_observed=False,pose_modified=False)
    y,x=np.where(obj&~human);base['visible_pixels']=int(len(x))
    if len(x)<min_pixels:return dict(base,informative=False,reason='insufficient_visible_pixels')
    pixels=np.c_[x,y].astype(float);mean=pixels.mean(0);_,s,vh=np.linalg.svd(pixels-mean,full_matrices=False)
    ratio=float(s[0]/max(s[1],1e-12));base['mask_axis_ratio']=ratio
    points=np.array([center-direction*length,center,center+direction*length])@pose[:3,:3].T+pose[:3,3]
    if np.any(points[:,2]<=.01):return dict(base,informative=False,reason='camera_plane')
    projected=points@k.T;uv=projected[:,:2]/projected[:,2:];delta=uv[2]-uv[0];span=float(np.linalg.norm(delta))
    if ratio<min_axis_ratio or span<2:return dict(base,informative=False,reason='foreshortened_or_nonelongated_view')
    cosine=float(np.clip(abs(delta/span@vh[0]),0,1))
    return dict(base,informative=True,axis_error_deg=float(np.degrees(np.arccos(cosine))),
        mask_center_xy=mean.tolist(),projected_axis_center_xy=uv[1].tolist(),
        center_error_px=float(np.linalg.norm(uv[1]-mean)),projected_axis_span_px=span)


def cross_section_audit(mask, depth_m, intrinsics, vertices, *, scale=1., bins=16,
                        warning_ratio=1.5):
    mask = np.asarray(mask, bool)
    depth_m = np.asarray(depth_m, float)
    k = np.asarray(intrinsics, float)
    vertices = np.asarray(vertices, float)
    if (mask.ndim != 2 or depth_m.shape != mask.shape or k.shape != (3, 3)
            or vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 3
            or not np.isfinite(vertices).all() or not np.isfinite(k).all()
            or not np.isfinite(depth_m).all() or not np.isfinite(scale)
            or scale <= 0 or bins < 4 or warning_ratio <= 1):
        raise ValueError('Invalid mask, depth, camera, mesh or audit bounds')
    if min(k[0, 0], k[1, 1]) <= 0:
        raise ValueError('Positive focal length required')
    y, x = np.where(mask)
    if len(x) < 100:
        raise ValueError('At least 100 visible object pixels required')
    observed_depth = depth_m[mask & (depth_m > 0)]
    if len(observed_depth) < 100:
        raise ValueError('At least 100 positive source-depth pixels required')
    pixels = np.stack((x, y), axis=1).astype(float)
    center = np.median(pixels, axis=0)
    _, _, directions = np.linalg.svd(pixels - center, full_matrices=False)
    long = (pixels - center) @ directions[0]
    across = (pixels - center) @ directions[1]
    edges = np.linspace(*np.quantile(long, [.01, .99]), bins + 1)
    local_widths = []
    for lower, upper in zip(edges[:-1], edges[1:]):
        values = across[(long >= lower) & (long < upper)]
        if len(values) >= 20:
            local_widths.append(float(np.quantile(values, .99) - np.quantile(values, .01)))
    if not local_widths:
        raise ValueError('Insufficient local mask cross sections')
    observed_width_px = max(local_widths)
    pixel_direction = directions[1]
    metres_per_pixel = float(np.linalg.norm(pixel_direction / np.diag(k)[:2])) * float(np.median(observed_depth))
    observed_width_m = observed_width_px * metres_per_pixel
    centered = vertices - np.mean(vertices, axis=0)
    _, _, mesh_axes = np.linalg.svd(centered[::max(1, len(centered) // 10000)], full_matrices=False)
    mesh_extents = sorted((float(np.ptp(centered @ axis)) * float(scale) for axis in mesh_axes), reverse=True)
    mesh_breadth_m = mesh_extents[1]
    ratio = float(observed_width_m / max(mesh_breadth_m, 1e-9))
    return dict(schema='real2sim2real.elongated-shape-audit/1',
        observed_max_local_width_px=observed_width_px,
        observed_local_width_m=observed_width_m,
        mesh_second_principal_extent_m=mesh_breadth_m,
        ratio=ratio, warning_ratio=float(warning_ratio),
        review_required=bool(ratio > warning_ratio),
        conclusion='Possible missing transverse geometry; inspect source RGB, mask and mesh before registration' if ratio > warning_ratio else 'No missing-crossbar flag from this view',
        source_depth_is_prediction=True, pose_validated=False, mesh_modified=False)
