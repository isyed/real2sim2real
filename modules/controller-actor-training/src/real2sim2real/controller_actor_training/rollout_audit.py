"""CPU-only pose and timing diagnostics for immutable saved rollouts.

Callers supply points, thresholds and clocks. Translation/rotation-only errors
are explanatory counterfactuals, never replacement trajectories or success
claims. Contact phases and motion-derived phases are explicitly distinguished.
"""
import numpy as np
from scipy.spatial.transform import Rotation


def poses(value):
    q=np.asarray(value,dtype=np.float64)
    if q.ndim!=2 or q.shape[1]!=7 or not len(q) or not np.isfinite(q).all():
        raise ValueError('Finite nonempty [frame,xyz+xyzw] poses required')
    norm=np.linalg.norm(q[:,3:],axis=1)
    if np.any(norm<1e-8):raise ValueError('Zero quaternion')
    q=q.copy();q[:,3:]/=norm[:,None]
    return q


def pose_errors(actual,target,points,*,chunk_size=64):
    a,b=poses(actual),poses(target);v=np.asarray(points,dtype=np.float64)
    if a.shape!=b.shape or v.ndim!=2 or v.shape[1]!=3 or not len(v) or not np.isfinite(v).all():
        raise ValueError('Matched poses and finite object-local points required')
    if type(chunk_size) is not int or chunk_size<=0:raise ValueError('Positive chunk size required')
    ra,rb=Rotation.from_quat(a[:,3:]).as_matrix(),Rotation.from_quat(b[:,3:]).as_matrix()
    delta=a[:,:3]-b[:,:3];add=np.empty(len(a));rotation_only=np.empty(len(a))
    for start in range(0,len(a),chunk_size):
        end=min(start+chunk_size,len(a))
        dv=np.einsum('tij,vj->tvi',ra[start:end]-rb[start:end],v)
        rotation_only[start:end]=np.linalg.norm(dv,axis=-1).mean(axis=-1)
        add[start:end]=np.linalg.norm(dv+delta[start:end,None,:],axis=-1).mean(axis=-1)
    angle=(Rotation.from_quat(a[:,3:])*Rotation.from_quat(b[:,3:]).inv()).magnitude()
    return dict(add_m=add,position_m=np.linalg.norm(delta,axis=1),rotation_only_add_m=rotation_only,
                orientation_deg=np.rad2deg(angle))


def add_summary(add,thresholds):
    x=np.asarray(add,dtype=float);t=np.asarray(thresholds,dtype=float)
    if x.ndim!=1 or not len(x) or not np.isfinite(x).all() or np.any(x<0):raise ValueError('Nonnegative finite ADD series required')
    if t.ndim!=1 or len(t)<2 or not np.isfinite(t).all() or np.any(t<=0) or np.any(np.diff(t)<=0):raise ValueError('Increasing positive thresholds required')
    accuracy=np.array([np.mean(x<limit) for limit in t])
    auc=float(np.trapezoid(accuracy,x=(t-t[0])/(t[-1]-t[0])))
    return dict(auc=auc,mean_add_m=float(x.mean()),median_add_m=float(np.median(x)),
        p95_add_m=float(np.percentile(x,95)),thresholds_m=t.tolist(),accuracy=accuracy.tolist(),
        fraction_above_max_threshold=float(np.mean(x>=t[-1])))


def motion_screen(reference,*,fps,translation_limit_m=.08,rotation_limit_deg=30.):
    q=poses(reference)
    if not np.isfinite(fps) or fps<=0:raise ValueError('Explicit positive native FPS required')
    delta=np.linalg.norm(np.diff(q[:,:3],axis=0),axis=1)
    angles=(Rotation.from_quat(q[1:,3:])*Rotation.from_quat(q[:-1,3:]).inv()).magnitude()*180/np.pi
    warnings=np.flatnonzero((delta>translation_limit_m)|(angles>rotation_limit_deg))+1
    return dict(translation_warning_frames=int(np.count_nonzero(delta>translation_limit_m)),
        rotation_warning_frames=int(np.count_nonzero(angles>rotation_limit_deg)),
        warning_frames=warnings.tolist(),first_translation_step_m=float(delta[0]) if len(delta) else 0.,
        median_translation_step_m=float(np.median(delta)) if len(delta) else 0.,
        first_rotation_step_deg=float(angles[0]) if len(angles) else 0.,
        max_translation_step_m=float(delta.max(initial=0)),
        max_rotation_step_deg=float(angles.max(initial=0)),translation_limit_m=translation_limit_m,
        rotation_limit_deg=rotation_limit_deg,scope='Native-frame motion screen, not a reconstruction acceptance gate')


def phase_labels(reference,contact,*,fps):
    """Evidence-derived phases; no assertion that the human completed a task."""
    q=poses(reference);c=np.asarray(contact,dtype=bool)
    if c.shape!=(len(q),):raise ValueError('One explicit contact-phase flag per native frame required')
    if not np.isfinite(fps) or fps<=0:raise ValueError('Explicit positive FPS required')
    velocity=np.r_[0,np.linalg.norm(np.diff(q[:,:3],axis=0),axis=1)*fps]
    angle=np.r_[0,(Rotation.from_quat(q[1:,3:])*Rotation.from_quat(q[:-1,3:]).inv()).magnitude()*fps]
    moving=(velocity>.02)|(angle>np.deg2rad(5))
    # Require sustained motion; one jump alone cannot label an entire transport phase.
    width=max(1,min(len(q),int(round(.2*fps))))
    moving=np.convolve(moving.astype(float),np.ones(width)/width,mode='same')>=.6
    events=np.flatnonzero(moving|c);labels=np.full(len(q),'stationary',dtype='<U24')
    if len(events):
        labels[:events[0]]='approach'
        labels[events[-1]+1:]='late_rest'
    labels[moving]='object_motion';labels[c]='labelled_contact'
    return labels


def intervals(flags,*,fps,minimum_seconds=0.,limit=10):
    f=np.asarray(flags,dtype=bool)
    if f.ndim!=1 or fps<=0:raise ValueError('Flags and positive FPS required')
    edges=np.diff(np.r_[False,f,False].astype(int));starts=np.flatnonzero(edges==1);ends=np.flatnonzero(edges==-1)
    rows=[dict(start_frame=int(a),end_frame_exclusive=int(b),start_s=float(a/fps),end_s=float(b/fps),duration_s=float((b-a)/fps))
          for a,b in zip(starts,ends) if (b-a)/fps>=minimum_seconds]
    return sorted(rows,key=lambda r:r['duration_s'],reverse=True)[:limit]


def phase_summary(errors,labels,thresholds):
    labels=np.asarray(labels)
    if labels.shape!=np.asarray(errors['add_m']).shape:raise ValueError('One phase per scored frame required')
    rows=[]
    for label in dict.fromkeys(labels.tolist()):
        mask=labels==label
        row=add_summary(np.asarray(errors['add_m'])[mask],thresholds)
        row.update(phase=label,frames=int(mask.sum()),fraction=float(mask.mean()),
            mean_position_m=float(np.asarray(errors['position_m'])[mask].mean()),
            mean_orientation_deg=float(np.asarray(errors['orientation_deg'])[mask].mean()))
        rows.append(row)
    return rows
