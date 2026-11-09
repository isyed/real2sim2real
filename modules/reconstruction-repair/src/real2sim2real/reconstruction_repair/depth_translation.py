"""Bounded camera-ray translation hypotheses from signed surface-depth error.

Monocular depth is a prediction. These proposals require independent native
image/depth, support and adjacent-frame checks; they do not fill hidden motion.
"""
import numpy as np


def finite_ray_proposals(vertices, pose, *, signed_distances_m, max_move_m=.15):
    """Explicit finite search along the mesh-centroid ray, in Euclidean metres.

    Negative distances approach the camera. This is separate from the signed
    residual heuristic: mixed-sign residuals can make that heuristic stop short.
    Candidates are hypotheses only. The caller must evaluate absolute depth
    discrepancy, image/support consistency and motion from accepted history.
    """
    from real2sim2real.reconstruction.recovery_quality import rigid
    v,p=np.asarray(vertices,float),np.asarray(pose,float)
    distances=np.asarray(signed_distances_m,float)
    if (v.ndim!=2 or v.shape[1]!=3 or not len(v) or not np.isfinite(v).all() or not rigid(p)
        or distances.ndim!=1 or not 1<=len(distances)<=64 or not np.isfinite(distances).all()
        or len(np.unique(distances))!=len(distances) or not np.isfinite(max_move_m)
        or not 0<max_move_m<=.25 or np.any(np.abs(distances)>max_move_m)):
        raise ValueError('Finite metric mesh, rigid pose and at most 64 unique bounded distances required')
    center=p[:3,:3]@v.mean(0)+p[:3,3]
    if center[2]<=.01:raise ValueError('Positive camera depth required')
    ray=center/np.linalg.norm(center);rows=[]
    for distance in distances:
        delta=ray*distance
        if center[2]+delta[2]<=.01:continue
        q=p.copy();q[:3,3]+=delta
        rows.append(dict(pose_camera=q.tolist(),signed_translation_m=float(distance),
            translation_change_m=float(abs(distance)),max_move_m=float(max_move_m),
            correction_bound_origin='raw_tracker_proposal',rotation_preserved=True,
            centroid_projection_preserved=True,mesh_scale_changed=False,depth_is_prediction=True,
            candidate_approved=False,native_image_depth_support_and_motion_checks_required=True))
    return rows


def bounded_proposals(vertices, pose, signed_surface_residual_m, *,
                      target_residual_m=.03, max_move_m=.15):
    """Three fixed-rotation hypotheses with a hard correction-distance cap.

    This cap is measured from the failed proposal. The caller must separately
    check motion from the last accepted state and all source quality gates.
    The original signed residual is retained even when the correction is capped.
    """
    from real2sim2real.reconstruction.recovery_quality import rigid
    v,p=np.asarray(vertices,float),np.asarray(pose,float)
    if (v.ndim!=2 or v.shape[1]!=3 or not len(v) or not np.isfinite(v).all() or not rigid(p)
        or not np.isfinite([signed_surface_residual_m,target_residual_m,max_move_m]).all()
        or not 0<target_residual_m<.1 or not 0<max_move_m<=.25):
        raise ValueError('Finite metric mesh, rigid pose and explicit correction bounds required')
    center=p[:3,:3]@v.mean(0)+p[:3,3]
    if center[2]<=.01:raise ValueError('Positive camera depth required')
    residual=float(signed_surface_residual_m)
    desired=center/center[2]*np.sign(residual)*max(0.,abs(residual)-target_residual_m)
    distance=float(np.linalg.norm(desired))
    if distance<1e-8:return []
    delta=desired*min(1.,max_move_m/distance)
    rows=[]
    for fraction in (.5,.75,1.):
        q=p.copy();q[:3,3]+=delta*fraction
        if center[2]+delta[2]*fraction<=.01:continue
        rows.append(dict(pose_camera=q.tolist(),fraction=fraction,
            signed_source_surface_residual_m=residual,target_residual_m=target_residual_m,
            desired_translation_change_m=distance,translation_change_m=float(np.linalg.norm(delta*fraction)),
            max_move_m=max_move_m,correction_capped=distance>max_move_m,
            rotation_preserved=True,centroid_projection_preserved=True,mesh_scale_changed=False,
            depth_is_prediction=True,candidate_approved=False))
    return rows


def propose(vertices,pose,signed_surface_residual_m,*,target_residual_m=.03,max_move_m=.15):
    from real2sim2real.reconstruction.recovery_quality import rigid
    v,p=np.asarray(vertices,float),np.asarray(pose,float)
    numbers=np.asarray([signed_surface_residual_m,target_residual_m,max_move_m],float)
    if (v.ndim!=2 or v.shape[1]!=3 or not len(v) or not np.isfinite(v).all() or not rigid(p)
        or not np.isfinite(numbers).all() or not 0<target_residual_m<.1 or not 0<max_move_m<=.25):
        raise ValueError('Finite metric mesh, rigid observed pose, signed residual and explicit small bounds required')
    center=p[:3,:3]@v.mean(0)+p[:3,3]
    if center[2]<=.01:raise ValueError('Positive camera-space centroid depth required')
    correction=np.sign(signed_surface_residual_m)*max(0.,abs(signed_surface_residual_m)-target_residual_m)
    delta=center/center[2]*correction
    if np.linalg.norm(delta)>max_move_m or center[2]+correction<=.01:
        raise ValueError('Observed-depth proposal exceeds declared metric bound')
    result=p.copy();result[:3,3]+=delta
    return result,dict(schema='real2sim2real.camera-ray-depth-hypothesis/1',
        signed_source_surface_residual_m=float(signed_surface_residual_m),target_residual_m=target_residual_m,
        camera_depth_change_m=float(correction),translation_change_m=float(np.linalg.norm(delta)),
        max_move_m=max_move_m,centroid_projection_preserved=True,rotation_preserved=True,
        mesh_scale_changed=False,interpolation_used=False,depth_is_prediction=True,
        candidate_approved=False,native_image_depth_support_and_motion_checks_required=True)
