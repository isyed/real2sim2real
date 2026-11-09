"""A ground-supported size/depth hypothesis that preserves image projection.

Scaling a metric mesh AND its camera-frame translations by the same positive
factor leaves pinhole projections unchanged. An independently predicted floor
and explicitly reviewed resting frames constrain that ambiguity. The floor is
still a prediction: this is not metric truth or acceptance of a trajectory.
"""
import numpy as np
from real2sim2real.reconstruction.recovery_quality import rigid


def propose(vertices, poses_camera, plane_camera, frame_ids, fit, *, resting_evidence,
            max_relative_change=.10, max_contact_error_m=.02, max_fit_ratio_spread=.03):
    v,p,plane=map(lambda x:np.asarray(x,dtype=float),(vertices,poses_camera,plane_camera))
    ids,fit=map(np.asarray,(frame_ids,fit))
    n=len(p)
    if (v.ndim!=2 or v.shape[1]!=3 or len(v)<4 or not np.isfinite(v).all()
        or p.shape!=(n,4,4) or n<6 or not all(rigid(x) for x in p)
        or plane.shape!=(4,) or not np.isfinite(plane).all()
        or not np.isclose(np.linalg.norm(plane[:3]),1.,atol=1e-6) or plane[3]<=0
        or ids.shape!=(n,) or ids.dtype.kind not in 'iu' or np.any(np.diff(ids)<=0)
        or fit.shape!=(n,) or fit.dtype.kind!='b' or min(int(fit.sum()),int((~fit).sum()))<3
        or not isinstance(resting_evidence,str) or not resting_evidence.strip()
        or not np.isfinite([max_relative_change,max_contact_error_m,max_fit_ratio_spread]).all()
        or not 0<max_relative_change<.5 or max_contact_error_m<=0 or not 0<max_fit_ratio_spread<.5):
        raise ValueError('Metric camera geometry, unit floor, reviewed resting evidence and predeclared fit/held-out frames required')
    minimum=[]
    for pose in p:
        points=v@pose[:3,:3].T+pose[:3,3]
        if points[:,2].min()<=.01:raise ValueError('Object crosses camera near plane')
        minimum.append(float((points@plane[:3]).min()))
    minimum=np.asarray(minimum)
    if np.any(minimum>=-1e-9):raise ValueError('Floor intersection cannot determine positive camera-ray scale')
    ratios=-plane[3]/minimum
    scale=float(np.median(ratios[fit]))
    clearance=scale*minimum+plane[3]
    spread=float(np.ptp(ratios[fit])/scale)
    reasons=[]
    if abs(scale-1)>max_relative_change:reasons.append('scale_bound')
    if spread>max_fit_ratio_spread:reasons.append('fit_ratio_disagreement')
    if np.max(np.abs(clearance[fit]))>max_contact_error_m:reasons.append('fit_ground_disagreement')
    if np.max(np.abs(clearance[~fit]))>max_contact_error_m:reasons.append('heldout_ground_disagreement')
    changed=p.copy();changed[:,:3,3]*=scale
    return v*scale,changed,dict(schema='real2sim2real.resting-ray-scale/1',accepted=not reasons,reasons=reasons,
        scale_multiplier=scale,fit_ratio_spread=spread,frame_ids=ids.tolist(),fit_frame_ids=ids[fit].tolist(),
        heldout_frame_ids=ids[~fit].tolist(),individual_contact_scales=ratios.tolist(),
        ground_clearance_before_m=(minimum+plane[3]).tolist(),ground_clearance_after_m=clearance.tolist(),
        resting_evidence=resting_evidence,rotations_preserved=True,pinhole_projection_preserved=True,
        mesh_and_translations_scaled_together=True,depth_model_validated=False,interaction_validated=False,
        complete_reference_approved=False,physics_validated=False,actor_training_approved=False,
        absolute_metric_accuracy_established=False,
        limitations=['Floor and resting phase must be independently supported by source evidence.',
            'Identical projection provides no independent evidence that depth or physical size is correct.',
            'Fresh native depth/contact/pose checks are required before using scaled geometry.'])
