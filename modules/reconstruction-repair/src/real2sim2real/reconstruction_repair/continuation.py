"""Join an observed tracking continuation without silently splicing bad branches.

Numeric guard only: poses are never interpolated, transformed or optimized.
Consumers must independently validate size, image/depth/support and provenance.
"""
import numpy as np
from scipy.spatial.transform import Rotation


def _poses(poses, ids):
    p, ids = np.asarray(poses), np.asarray(ids)
    if (ids.ndim != 1 or ids.dtype.kind not in 'iu' or len(ids)<2 or np.any(np.diff(ids)!=1)
            or p.shape != (len(ids),4,4) or not np.isfinite(p).all()
            or not np.allclose(p[:,3],[0,0,0,1],atol=1e-7)
            or not np.allclose(p[:,:3,:3]@p[:,:3,:3].swapaxes(1,2),np.eye(3),atol=1e-4)
            or not np.allclose(np.linalg.det(p[:,:3,:3]),1,atol=1e-4)):
        raise ValueError('Continuation requires finite rigid poses on consecutive original frame IDs')
    return p, ids


def continuity(poses, ids, *, max_step_m=.08, max_rotation_deg=30.):
    p, ids = _poses(poses,ids)
    if not np.isfinite([max_step_m,max_rotation_deg]).all() or min(max_step_m,max_rotation_deg)<=0:
        raise ValueError('Positive finite continuity bounds are required')
    step=np.linalg.norm(np.diff(p[:,:3,3],axis=0),axis=1)
    angle=np.rad2deg((Rotation.from_matrix(p[1:,:3,:3])*Rotation.from_matrix(p[:-1,:3,:3]).inv()).magnitude())
    bad=(step>max_step_m+1e-6)|(angle>max_rotation_deg+1e-3)
    return dict(passed=not bool(bad.any()),max_translation_step_m=float(step.max()),
        max_rotation_step_deg=float(angle.max()),translation_max_frame=int(ids[step.argmax()+1]),
        rotation_max_frame=int(ids[angle.argmax()+1]),failed_frames=ids[1:][bad].tolist(),
        bounds=dict(max_step_m=max_step_m,max_rotation_deg=max_rotation_deg),
        limitation='Continuity does not establish accurate images/depth, contacts or task success.')


def merge_continuation(parent, parent_ids, tail, tail_ids, *, max_step_m=.08,max_rotation_deg=30.):
    p, ids = _poses(parent,parent_ids); t, tid = _poses(tail,tail_ids)
    if tid[-1]!=ids[-1] or tid[0]<ids[0] or tid[0]>=ids[-1]:
        raise ValueError('Continuation must cover the complete remaining suffix including its anchor')
    start=int(np.searchsorted(ids,tid[0]))
    if not np.array_equal(ids[start:],tid):raise ValueError('Continuation frame identities do not match parent suffix')
    if not np.allclose(p[start],t[0],rtol=0,atol=1e-6):
        raise ValueError('Continuation anchor disagrees with the last accepted parent pose')
    result=p.copy();result[start+1:]=t[1:]
    audit=continuity(result,ids,max_step_m=max_step_m,max_rotation_deg=max_rotation_deg)
    audit.update(anchor_frame=int(tid[0]),prefix_preserved_exactly=bool(np.array_equal(result[:start+1],p[:start+1])),
        interpolated_frames=0,pose_frame_transforms_applied=False,full_reference_accepted=False,
        physics_validated=False,approved_for_use=False)
    if not audit['passed']:
        raise ValueError('Merged continuation exceeds declared motion bounds at frames '+str(audit['failed_frames']))
    return result,audit
