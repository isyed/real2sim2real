"""Separate reference-relative palm tracking from measured contact retention.

Reference palm anchors are intended targets, not independently verified handle
locations. Physical contacts can be impacts; neither contact counts nor these
diagnostics certify a grasp. All poses use metres and xyzw quaternions.
"""
import numpy as np

def contact_intervals(contact, *, fps):
    a=np.asarray(contact)
    if a.ndim!=1 or a.dtype!=bool or not np.isfinite(fps) or fps<=0:
        raise ValueError('Boolean contact timeline and positive recording rate required')
    edges=np.diff(np.r_[False,a,False].astype(int));starts=np.flatnonzero(edges==1);ends=np.flatnonzero(edges==-1)
    return [dict(start_frame=int(s),end_frame_exclusive=int(e),frames=int(e-s),duration_s=float((e-s)/fps)) for s,e in zip(starts,ends)]

def palm_tracking(actual_palm,reference_palm,actual_object,reference_object,phase,contact,*,fps):
    from scipy.spatial.transform import Rotation
    poses=[np.asarray(x,float) for x in [actual_palm,reference_palm,actual_object,reference_object]]
    n=len(poses[0]);phase=np.asarray(phase);contact=np.asarray(contact)
    if (not n or any(x.shape!=(n,7) or not np.isfinite(x).all() or
            not np.allclose(np.linalg.norm(x[:,3:],axis=1),1,atol=1e-4) for x in poses)
            or phase.shape!=(n,) or phase.dtype!=bool or contact.shape!=(n,) or contact.dtype!=bool):
        raise ValueError('Aligned finite rigid xyzw poses and boolean phase/contact timelines required')
    ap,rp,ao,ro=poses
    Rap,Rrp,Rao,Rro=[Rotation.from_quat(x[:,3:]) for x in poses]
    local=Rro.inv().apply(rp[:,:3]-ro[:,:3]);wanted=Rao.apply(local)+ao[:,:3]
    position=np.linalg.norm(ap[:,:3]-wanted,axis=1)
    orientation=(Rap*(Rao*Rro.inv()*Rrp).inv()).magnitude()
    intervals=contact_intervals(contact & phase,fps=fps)
    physical_intervals=contact_intervals(contact,fps=fps)
    active=np.flatnonzero(phase)
    value=dict(active_frames=len(active),physical_contact_frames=int(contact.sum()),
        contact_during_phase_frames=int((contact&phase).sum()),contact_intervals=intervals,
        longest_contact_s=max([x['duration_s'] for x in intervals],default=0.),
        physical_contact_intervals=physical_intervals,
        longest_physical_contact_s=max([x['duration_s'] for x in physical_intervals],default=0.),
        median_position_error_m=float(np.median(position[phase])) if len(active) else None,
        median_orientation_error_deg=float(np.degrees(np.median(orientation[phase]))) if len(active) else None,
        active_frames_within_5cm=int((phase&(position<=.05)).sum()),
        reference_anchor_is_verified_handle=False,grasp_success_established=False)
    return value,dict(target_palm_position=wanted,position_error_m=position,orientation_error_rad=orientation,
        phase=phase,physical_contact=contact)
