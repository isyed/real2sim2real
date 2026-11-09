"""Conservative diagnostic assembly of independently checked pose segments.

The caller verifies native image/depth evidence and its hashes. This module
preserves the trusted baseline, never fills gaps, and screens newly created
motion joins. It does not approve a physical reference or certify source depth.
"""
import re
import numpy as np
from scipy.spatial.transform import Rotation

def _rigid(poses):
    p=np.asarray(poses)
    return (p.ndim==3 and p.shape[1:]==(4,4) and np.isfinite(p).all()
        and np.allclose(p[:,3],[0,0,0,1],atol=1e-5)
        and np.allclose(p[:,:3,:3]@p[:,:3,:3].transpose(0,2,1),np.eye(3),atol=1e-4)
        and np.allclose(np.linalg.det(p[:,:3,:3]),1.,atol=1e-4))

def merge(baseline_poses, baseline_available, segments, *, max_translation_m=.08,
          max_rotation_deg=30.):
    """Fill unavailable slots, dropping new boundary poses with invalid joins.

    Segments are ordered by priority and declare ``source_frame_ids``,
    ``poses_camera``, ``evidence_sha256`` and ``native_validation_passed=True``.
    Native IDs may run forward or backward but must be contiguous. Overlap
    never overwrites a trusted baseline pose. Data declarations are not proof
    of validation: the caller owns matching the receipt to actual source data.
    """
    poses=np.asarray(baseline_poses);locked=np.asarray(baseline_available)
    if (poses.ndim!=3 or not len(poses) or poses.shape[1:]!=(4,4) or locked.dtype.kind!='b'
            or locked.shape!=(len(poses),) or not np.isfinite(poses).all()
            or not _rigid(poses[locked]) or not np.isfinite([max_translation_m,max_rotation_deg]).all()
            or isinstance(max_translation_m,bool) or isinstance(max_rotation_deg,bool)
            or max_translation_m<=0 or not 0<max_rotation_deg<=180):
        raise ValueError('Finite pose timeline, boolean availability and positive motion bounds required')
    result=poses.copy();available=locked.copy();origin=np.full(len(poses),-1,int);records=[]
    def step(a,b):
        return float(np.linalg.norm(b[:3,3]-a[:3,3])),float(np.rad2deg(Rotation.from_matrix(b[:3,:3]@a[:3,:3].T).magnitude()))
    def invalid(a,b):
        d,r=step(a,b)
        return d>max_translation_m+1e-9 or r>max_rotation_deg+1e-6,d,r
    for i in np.flatnonzero(locked[:-1]&locked[1:]):
        if invalid(result[i],result[i+1])[0]:raise ValueError('Trusted baseline already has an invalid motion join')
    for priority,segment in enumerate(segments):
        ids=np.asarray(segment['source_frame_ids']);p=np.asarray(segment['poses_camera'])
        evidence=segment.get('evidence_sha256','')
        if (ids.ndim!=1 or not len(ids) or ids.dtype.kind not in 'iu' or np.any(ids<0)
                or np.any(ids>=len(poses)) or p.shape!=(len(ids),4,4) or not _rigid(p)
                or len(ids)>1 and not (np.all(np.diff(ids.astype(np.int64))==1) or np.all(np.diff(ids.astype(np.int64))==-1))
                or segment.get('native_validation_passed') is not True
                or not isinstance(evidence,str) or re.fullmatch('[0-9a-f]{64}',evidence) is None):
            raise ValueError('Contiguous native segment with rigid poses and explicit validation receipt required')
        for a,b in zip(p[:-1],p[1:]):
            if invalid(a,b)[0]:raise ValueError('Offered segment itself has an invalid motion step')
        take=~available[ids];new=ids[take];result[new]=p[take];available[new]=True;origin[new]=priority
        records.append(dict(name=str(segment.get('name',priority)),evidence_sha256=evidence,
            offered_frames=len(ids),new_frames=int(take.sum()),overlap_preserved=int((~take).sum())))
    rejected=[]
    for i in range(len(poses)-1):
        if not (available[i] and available[i+1]):continue
        bad,d,r=invalid(result[i],result[i+1])
        if not bad:continue
        if locked[i] and locked[i+1]:raise ValueError('Trusted baseline was changed')
        drop=i+1 if locked[i] else i if locked[i+1] else (i+1 if origin[i+1]>=origin[i] else i)
        available[drop]=False
        rejected.append(dict(source_frames=[i,i+1],dropped_new_frame=int(drop),translation_m=d,rotation_deg=r))
    if not np.array_equal(result[locked],poses[locked]) or not available[locked].all():raise AssertionError('Trusted baseline changed')
    result[~available]=poses[~available]
    report=dict(schema='real2sim2real.pose-segment-assembly/1',baseline_frames=int(locked.sum()),
        available_frames=int(available.sum()),new_frames=int(available.sum()-locked.sum()),segments=records,
        rejected_joins=rejected,baseline_preserved_exactly=True,gaps_filled=False,
        max_translation_m=float(max_translation_m),max_rotation_deg=float(max_rotation_deg),
        join_roundoff_tolerance=dict(translation_m=1e-9,rotation_deg=1e-6),
        full_timeline_available=bool(available.all()),physical_reference_approved=False,
        caller_owned_native_evidence=True)
    return result,available,report
