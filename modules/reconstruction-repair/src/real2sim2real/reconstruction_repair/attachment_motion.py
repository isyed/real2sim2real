"""Diagnose a rigid object/carrier attachment across unavailable observations.

Carrier landmarks may be predicted palm vertices. Motion is then INFERRED from
that contact assumption, not measured through occlusion. Independent object
anchors can veto the hypothesis; no reference or physical approval is returned.
"""
import numpy as np
from scipy.spatial.transform import Rotation


def _poses(value):
    p=np.asarray(value,float)
    if (p.ndim!=3 or p.shape[1:]!=(4,4) or not len(p) or not np.isfinite(p).all()
        or not np.allclose(p[:,3],[0,0,0,1],atol=1e-7)
        or not np.allclose(p[:,:3,:3]@p[:,:3,:3].swapaxes(1,2),np.eye(3),atol=1e-5)
        or not np.allclose(np.linalg.det(p[:,:3,:3]),1,atol=1e-5)):
        raise ValueError('Finite metric rigid poses required')
    return p


def carrier_frames(points,*,reference_frame=0,max_rms_m=.01):
    """Fit proper rigid frames from corresponding anatomical points, in metres."""
    p=np.asarray(points,float)
    if (p.ndim!=3 or p.shape[2]!=3 or p.shape[1]<3 or not np.isfinite(p).all()
        or type(reference_frame) is not int or not 0<=reference_frame<len(p)
        or not np.isfinite(max_rms_m) or max_rms_m<=0):raise ValueError('Corresponding metric points and a valid reference frame required')
    centers=p.mean(1);centered=p-centers[:,None];ref=centered[reference_frame]
    if np.linalg.matrix_rank(ref,tol=1e-8)<2:raise ValueError('Carrier points must not be collinear')
    poses=np.tile(np.eye(4),(len(p),1,1));rms=[]
    for i,cloud in enumerate(centered):
        if np.linalg.matrix_rank(cloud,tol=1e-8)<2:raise ValueError('Degenerate carrier frame')
        u,_,vh=np.linalg.svd(ref.T@cloud);r=vh.T@u.T
        if np.linalg.det(r)<0:vh[-1]*=-1;r=vh.T@u.T
        poses[i,:3,:3]=r;poses[i,:3,3]=centers[i]
        rms.append(float(np.sqrt(np.mean(np.sum((ref@r.T-cloud)**2,axis=1)))))
    return poses,dict(rms_m=rms,maximum_rms_m=max(rms),valid_frames=[i for i,x in enumerate(rms) if x<=max_rms_m],
        max_rms_m=max_rms_m,proper_rotation=True,scale_fitted=False,contact_proven=False)


def diagnose(carriers,observations,*,fit_frames,heldout_frames,attachment_evidence,
             max_translation_residual_m=.10,max_rotation_residual_deg=20.,rotation_observed_frames=None):
    """Fit only declared anchors; a disjoint set can only veto that attachment.

    ``observations`` maps source frame IDs to independently estimated object
    poses. The caller retains provenance and validation type for those anchors.
    A pass diagnoses consistency only; hidden motion remains unobserved.
    """
    c=_poses(carriers);fit=list(fit_frames);held=list(heldout_frames)
    if (len(fit)<3 or len(held)<2 or len(set(fit))!=len(fit) or len(set(held))!=len(held)
        or set(fit)&set(held) or any(type(i) is not int or not 0<=i<len(c) for i in fit+held)
        or not isinstance(attachment_evidence,str) or not attachment_evidence.strip()
        or not np.isfinite([max_translation_residual_m,max_rotation_residual_deg]).all()
        or min(max_translation_residual_m,max_rotation_residual_deg)<=0):
        raise ValueError('Distinct predeclared fit/held-out anchors and explicit contact evidence required')
    observed={i:_poses(np.asarray(observations[i])[None])[0] for i in fit+held}
    rotation_frames=set(fit+held if rotation_observed_frames is None else rotation_observed_frames)
    if not set(fit)<=rotation_frames or not rotation_frames<=set(fit+held):
        raise ValueError('Rigid fitting requires independent fit rotations; validation rotation scope must match anchors')
    relative=np.stack([np.linalg.inv(c[i])@observed[i] for i in fit])
    offset=np.eye(4);offset[:3,:3]=Rotation.from_matrix(relative[:,:3,:3]).mean().as_matrix()
    offset[:3,3]=np.median(relative[:,:3,3],axis=0)
    predicted=c@offset
    rows=[]
    for i in fit+held:
        dt=float(np.linalg.norm(predicted[i,:3,3]-observed[i][:3,3]))
        angle=float(np.rad2deg(Rotation.from_matrix(predicted[i,:3,:3]@observed[i][:3,:3].T).magnitude()))
        checked=i in rotation_frames
        rows.append(dict(frame=i,split='fit' if i in fit else 'held_out',translation_residual_m=dt,
            rotation_residual_deg=angle if checked else None,raw_rotation_comparison_deg=angle,
            rotation_independently_checked=checked,
            passed=dt<=max_translation_residual_m and (not checked or angle<=max_rotation_residual_deg)))
    report=dict(schema='real2sim2real.attachment-motion/1',fit_frames=fit,heldout_frames=held,
        relative_object_pose=offset.tolist(),residuals=rows,anchor_consistency_passed=all(r['passed'] for r in rows),
        fit_score=float(np.mean([r['translation_residual_m']/max_translation_residual_m+r['rotation_residual_deg']/max_rotation_residual_deg for r in rows if r['split']=='fit'])),
        bounds=dict(translation_m=max_translation_residual_m,rotation_deg=max_rotation_residual_deg),
        attachment_evidence=attachment_evidence,heldout_used_for_fitting=False,
        rotation_observed_frames=sorted(rotation_frames),unvalidated_rotation_frames=sorted(set(fit+held)-rotation_frames),
        inferred_motion=True,hidden_motion_measured=False,reference_approved=False,physics_validated=False,
        limitation='A stable carrier attachment is an assumed model. Agreement with sparse predicted anchors cannot establish true hidden motion.')
    return predicted,report


def diagnose_point(carriers,observed_positions,*,fit_frames,heldout_frames,attachment_evidence,
                   max_residual_m=.10):
    """Object-origin point attached to a carrier; object orientation is FREE.

    A changing wrist/object rotation need not invalidate a positional contact
    constraint. This API deliberately returns positions only, never orientations.
    """
    c=_poses(carriers);fit=list(fit_frames);held=list(heldout_frames)
    if (len(fit)<3 or len(held)<2 or set(fit)&set(held) or len(set(fit+held))!=len(fit+held)
        or any(type(i) is not int or not 0<=i<len(c) for i in fit+held)
        or not isinstance(attachment_evidence,str) or not attachment_evidence.strip()
        or not np.isfinite(max_residual_m) or max_residual_m<=0):raise ValueError('Explicit disjoint fit/validation anchors required')
    obs={i:np.asarray(observed_positions[i],float) for i in fit+held}
    if any(x.shape!=(3,) or not np.isfinite(x).all() for x in obs.values()):raise ValueError('Metric position anchors required')
    offset=np.median(np.stack([c[i,:3,:3].T@(obs[i]-c[i,:3,3]) for i in fit]),axis=0)
    predicted=np.einsum('tij,j->ti',c[:,:3,:3],offset)+c[:,:3,3]
    rows=[dict(frame=i,split='fit' if i in fit else 'held_out',residual_m=float(np.linalg.norm(predicted[i]-obs[i]))) for i in fit+held]
    report=dict(schema='real2sim2real.point-attachment/1',fit_frames=fit,heldout_frames=held,
        carrier_local_offset_m=offset.tolist(),residuals=rows,bound_m=max_residual_m,
        anchor_consistency_passed=all(x['residual_m']<=max_residual_m for x in rows),
        attachment_evidence=attachment_evidence,heldout_used_for_fitting=False,
        object_orientation_estimated=False,hidden_motion_measured=False,reference_approved=False)
    return predicted,report


def bridge_positions(predicted,*,first_frame,last_frame,first_position,last_position,
                     max_endpoint_correction_m=.10,max_step_m=.08):
    """Carrier-driven position hypothesis with bounded, exact endpoint anchors.

    Endpoint discrepancy is spread smoothly, while the carrier supplies the
    interior motion. Outputs remain INFERRED even if numeric bounds pass.
    No source observations are changed and no orientation is synthesized.
    """
    p=np.asarray(predicted,float);a=np.asarray(first_position,float);b=np.asarray(last_position,float)
    if (p.ndim!=2 or p.shape[1]!=3 or not np.isfinite(p).all() or a.shape!=(3,) or b.shape!=(3,)
        or not np.isfinite([*a,*b,max_endpoint_correction_m,max_step_m]).all()
        or min(max_endpoint_correction_m,max_step_m)<=0 or type(first_frame) is not int or type(last_frame) is not int
        or not 0<=first_frame<last_frame<len(p)):raise ValueError('Finite metric positions and ordered endpoint frames required')
    delta_a=a-p[first_frame];delta_b=b-p[last_frame]
    t=np.linspace(0,1,last_frame-first_frame+1);weight=t*t*(3-2*t)
    correction=(1-weight[:,None])*delta_a+weight[:,None]*delta_b
    proposal=p[first_frame:last_frame+1]+correction;proposal[0]=a;proposal[-1]=b
    step=float(np.linalg.norm(np.diff(proposal,axis=0),axis=1).max())
    adjustment=float(np.linalg.norm(correction,axis=1).max());reasons=[]
    if adjustment>max_endpoint_correction_m:reasons.append('endpoint_correction_bound')
    if step>max_step_m:reasons.append('position_step_bound')
    return proposal,dict(schema='real2sim2real.carrier-position-bridge/1',first_frame=first_frame,last_frame=last_frame,
        max_correction_m=adjustment,max_step_m=step,passed_numeric_bounds=not reasons,reasons=reasons,
        bounds=dict(endpoint_correction_m=max_endpoint_correction_m,step_m=max_step_m),
        endpoints_preserved_exactly=True,inferred_frame_ids=list(range(first_frame+1,last_frame)),
        object_orientation_estimated=False,hidden_motion_measured=False,reference_approved=False)


def regularize_inferred_positions(positions,*,sigma_frames=1.,max_correction_m=.03,max_step_m=.08):
    """One bounded smoothing proposal for INFERRED positions, never observations.

    A Gaussian proposal is projected into a per-frame correction ball. Endpoint
    anchors remain exact. Numeric limits can veto it; no iterative gate tuning.
    """
    from scipy.ndimage import gaussian_filter1d
    p=np.asarray(positions,float)
    if (p.ndim!=2 or p.shape[1]!=3 or len(p)<3 or not np.isfinite(p).all()
        or not np.isfinite([sigma_frames,max_correction_m,max_step_m]).all()
        or min(sigma_frames,max_correction_m,max_step_m)<=0):raise ValueError('Finite inferred positions and positive bounds required')
    q=gaussian_filter1d(p,sigma_frames,axis=0,mode='nearest');t=np.linspace(0,1,len(p))[:,None]
    q+=(1-t)*(p[0]-q[0])+t*(p[-1]-q[-1])
    delta=q-p;norm=np.linalg.norm(delta,axis=1)
    delta*=np.minimum(1.,max_correction_m/np.maximum(norm,1e-15))[:,None]
    q=p+delta;q[0]=p[0];q[-1]=p[-1]
    step=float(np.linalg.norm(np.diff(q,axis=0),axis=1).max())
    return q,dict(schema='real2sim2real.inferred-position-regularization/1',sigma_frames=sigma_frames,
        maximum_correction_m=float(np.linalg.norm(q-p,axis=1).max()),maximum_step_m=step,
        bounds=dict(correction_m=max_correction_m,step_m=max_step_m),passed_numeric_bounds=step<=max_step_m+1e-9,
        endpoints_preserved_exactly=True,source_observations_modified=False,hidden_motion_measured=False,reference_approved=False)
