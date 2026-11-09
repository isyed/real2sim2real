"""Bounded two-frame look-back proposals; every old and new gate stays active."""
from dataclasses import asdict
import numpy as np
from real2sim2real.reconstruction.recovery_quality import (
    Limits,rigid,sample_vertices,consistency,rejection_reasons,
)
from .geometry import mesh_silhouette
from .local_registration import propose,LocalRegistrationConfig


def propose_pair(vertices,faces,previous,anchor,camera,objects,occluders,depths,support,*,
                 source_frame_ids,ray_shifts_m=(-.02,-.04,-.06),
                 config=LocalRegistrationConfig(depth_violation_weight=2.)):
    """Recheck a recent anchor and recover its next frame, without changing inputs.

    ``previous`` belongs to source_frame_ids[0]-1. Both image/depth grids and
    poses use the same camera frame. Camera-ray shifts are explicit proposals,
    not inferred motion. Return no pair unless both native CPU checks and both
    temporal joins pass. An actual GPU revalidation is still required.
    """
    v,f,p,a,k=map(np.asarray,(vertices,faces,previous,anchor,camera))
    objects,occluders=np.asarray(objects,bool),np.asarray(occluders,bool)
    depths=np.asarray(depths,float);ids=np.asarray(source_frame_ids);shifts=np.asarray(ray_shifts_m,float)
    config.validate()
    if (not rigid(p) or not rigid(a) or a[2,3]<=.01 or k.shape!=(3,3)
        or not np.isfinite(k).all() or min(k[0,0],k[1,1])<=0 or not np.allclose(k[2],[0,0,1])
        or ids.shape!=(2,) or ids.dtype.kind not in 'iu' or ids[0]<1 or ids[1]!=ids[0]+1
        or objects.ndim!=3 or len(objects)!=2 or occluders.shape!=objects.shape or depths.shape!=objects.shape
        or not np.isfinite(depths).all() or shifts.ndim!=1 or not 1<=len(shifts)<=6
        or not np.isfinite(shifts).all() or np.any(np.abs(shifts)>.06)):
        raise ValueError('Two consecutive native frames, common camera and bounded finite shifts required')
    points=sample_vertices(v);rows=[];selected=None
    for shift in shifts:
        candidate=a.astype(float).copy();candidate[:3,3]+=a[:3,3]/a[2,3]*shift
        change=float(np.linalg.norm(candidate[:3,3]-a[:3,3]))
        row=dict(camera_depth_shift_m=float(shift),anchor_translation_change_m=change)
        rows.append(row)
        if change>.06:
            row['anchor_rejected_for']=['anchor_change_bound'];continue
        g=dict(vertices=v,faces=f,object_scale=np.array(1.),object_rotation=candidate[None,:3,:3],
            object_translation=candidate[None,:3,3],intrinsics=k[None],world_to_camera=np.eye(4)[None])
        observed=objects[0]&~occluders[0];mask=mesh_silhouette(g,0,*observed.shape)&~occluders[0]
        inter=int((mask&observed).sum());metrics=dict(iou=inter/max(1,int((mask|observed).sum())),
            precision=inter/max(1,int(mask.sum())),recall=inter/max(1,int(observed.sum())))
        geometry=consistency(points,candidate,k,depths[0],observed,support)
        rejected=rejection_reasons(candidate,p,metrics,geometry,recovery=True)
        row.update(anchor_pose=candidate.tolist(),anchor_metrics=metrics,anchor_geometry=geometry,anchor_rejected_for=rejected)
        if rejected:continue
        try:
            fitted=propose(v,f,candidate,candidate,k,objects[1],occluders[1],depths[1],support,config=config)
        except ValueError as error:
            row['next_frame_error']=str(error);continue
        row['next_frame_fit']=fitted
        if fitted['proposal_valid_for_tracking']:
            selected=np.stack([candidate,np.asarray(fitted['pose_camera'])]);break
    return selected,dict(schema='real2sim2real.lookback-recovery/1',source_frame_ids=ids.tolist(),
        preceding_source_frame=int(ids[0]-1),rows=rows,all_cpu_gates_passed=selected is not None,
        limits=asdict(Limits()),config=asdict(config),maximum_anchor_translation_m=.06,
        prior_prefix_before_anchor_preserved=True,gpu_revalidation_required=True,
        gpu_revalidation_source_frame_ids=ids.tolist(),
        continuation_seed_source_frame_after_both_gpu_checks=int(ids[-1]),
        thresholds_relaxed=False,full_reference_accepted=False,physics_validated=False)
