"""Small fixed-rotation translation hypotheses with exact native CPU checks.

Use after a complete observed pose sequence, not to fill missing frames. Native
depth remains a prediction. A passing CPU proposal still needs independent GPU
validation and the caller's source appearance and sequence checks.
"""
import time
import numpy as np
from .source_pose_audit import projection_audit


def propose(vertices, faces, pose, intrinsics, object_mask, human_mask, depth_m,
            previous, following, *, minimum_iou=.60, max_move_m=.01,
            max_seconds=90., depth_margin_m=.002,silhouette_scope='full',
            support_plane_camera=None,max_support_penetration_m=.02,
            require_sparse_native_depth=False):
    from real2sim2real.reconstruction.recovery_quality import rigid
    p,k,prev,nxt=map(lambda x:np.asarray(x,float),(pose,intrinsics,previous,following))
    if not all(rigid(x) for x in (p,prev,nxt)) or k.shape!=(3,3) or not np.isfinite(k).all() or p[2,3]<=0:
        raise ValueError('Rigid observed pose and both actual native neighbours required')
    if (not np.isfinite(minimum_iou) or not .60<=minimum_iou<=1 or
            not np.isfinite(max_move_m) or not 0<max_move_m<=.02 or
            not np.isfinite(max_seconds) or max_seconds<=0 or
            not np.isfinite(depth_margin_m) or not 0<depth_margin_m<.01
            or silhouette_scope not in ('full','human_excluded') or type(require_sparse_native_depth)!=bool):
        raise ValueError('Explicit small metric/time bounds and unchanged native limits required')
    plane=None if support_plane_camera is None else np.asarray(support_plane_camera,float)
    if plane is not None and (plane.shape!=(4,) or not np.isfinite(plane).all()
            or not np.isclose(np.linalg.norm(plane[:3]),1,atol=1e-6)
            or not np.isfinite(max_support_penetration_m) or max_support_penetration_m<0):
        raise ValueError('Unit metric support plane and nonnegative penetration bound required')
    if require_sparse_native_depth and plane is None:raise ValueError('Sparse native depth qualification requires explicit support geometry')
    vertices=np.asarray(vertices,float);started=time.monotonic();evaluations=0;tested=[]
    before,_=projection_audit(vertices@p[:3,:3].T+p[:3,3],faces,k,object_mask,depth_m,human_mask=human_mask)
    signed=before['predicted_surface_depth']['median_signed_m']
    need_depth=not before['predicted_surface_depth']['consistency_passed']
    sparse_before=None
    if require_sparse_native_depth:
        from real2sim2real.reconstruction.recovery_quality import consistency,sample_vertices
        points=sample_vertices(vertices)
        support=dict(plane_camera=plane.tolist(),center_camera_m=(-plane[:3]*plane[3]).tolist(),radius_m=1e6)
        sparse_before=consistency(points,p.astype(np.float32),k,np.asarray(depth_m,np.float32),
            np.asarray(object_mask,bool)&~np.asarray(human_mask,bool),support)
        if sparse_before['depth_points']<100:raise ValueError('Insufficient sparse native depth evidence')
        if sparse_before['depth_error_m']>.10:
            need_depth=True;signed=sparse_before['depth_shift_m']
    # Projective XY corrections use calibrated rays; Z shifts are explicit metres.
    depths=[0.]
    if need_depth:
        if signed is None:raise ValueError('Missing native predicted-depth evidence')
        direction=np.sign(signed)
        depths=[direction*x for x in [depth_margin_m,2*depth_margin_m,3*depth_margin_m]]
    def evaluate(dx,dy,dz):
        nonlocal evaluations
        if time.monotonic()-started>max_seconds:raise TimeoutError('Native translation proposal bound reached')
        delta=np.linalg.solve(k,np.array([dx*p[2,3],dy*p[2,3],0.]))+np.array([0.,0.,dz])
        q=p.copy();q[:3,3]+=delta
        if require_sparse_native_depth:q[:3,3]=q[:3,3].astype(np.float32)
        if np.linalg.norm(delta)>max_move_m or max(np.linalg.norm(q[:3,3]-prev[:3,3]),np.linalg.norm(q[:3,3]-nxt[:3,3]))>.08:
            return None
        camera=vertices@q[:3,:3].T+q[:3,3]
        if plane is not None and (camera@plane[:3]+plane[3]).min()<-max_support_penetration_m:return None
        result,_=projection_audit(camera,faces,k,object_mask,depth_m,human_mask=human_mask)
        evaluations+=1
        metric=result['full_silhouette' if silhouette_scope=='full' else 'human_excluded_silhouette']
        passing=(metric['iou']>=minimum_iou and result['predicted_surface_depth']['consistency_passed'])
        sparse=None
        if require_sparse_native_depth:
            sparse=consistency(points,q.astype(np.float32),k,np.asarray(depth_m,np.float32),
                np.asarray(object_mask,bool)&~np.asarray(human_mask,bool),support)
            passing=bool(passing and sparse['depth_points']>=100 and sparse['depth_error_m']<=.10
                and sparse['support_penetration_m']<=max_support_penetration_m and not sparse['behind_camera'])
        row=dict(translation_delta_m=delta.tolist(),offset_pixel=[dx,dy],camera_depth_shift_m=dz,native=result,sparse_native=sparse,passed=passing)
        tested.append(row)
        if passing:return dict(pose_camera=q.tolist(),before=before,selected=row,evaluations=evaluations,
            rotation_preserved=True,max_move_m=max_move_m,minimum_iou=minimum_iou,silhouette_scope=silhouette_scope,
            support_plane_checked=plane is not None,
            sparse_native_depth_required=require_sparse_native_depth,sparse_before=sparse_before,
            thresholds_relaxed=False,requires_gpu_validation=True,full_reference_accepted=False)
        return None
    # Start with the smallest change. Search is finite and stops at a valid fit.
    for dz in depths:
        result=evaluate(0.,0.,dz)
        if result:return result
    for radius in [.25,.5,1.,1.5,2.]:
        for dx,dy in [(radius,0),(-radius,0),(0,radius),(0,-radius),(radius,radius),(-radius,-radius),(radius,-radius),(-radius,radius)]:
            for dz in depths:
                result=evaluate(dx,dy,dz)
                if result:return result
    return dict(pose_camera=p.tolist(),before=before,tested=tested,evaluations=evaluations,
        passed=False,rotation_preserved=True,silhouette_scope=silhouette_scope,
        sparse_native_depth_required=require_sparse_native_depth,sparse_before=sparse_before,
        support_plane_checked=plane is not None,requires_gpu_validation=True,full_reference_accepted=False)
