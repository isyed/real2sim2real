"""Provenance-matched choice of the existing tracking/recovery image gate.

Uncovered, recovered or altered poses keep the stricter recovery gate. This
does not accept poses or change the shared image/depth/support/motion limits.
"""
import json
from pathlib import Path
import numpy as np
from real2sim2real.artifacts import digest


def recovery_flags(source_ids, poses, evidence_path, *, source_video_sha256):
    ids=np.asarray(source_ids);q=np.asarray(poses)
    if ids.ndim!=1 or ids.dtype.kind not in 'iu' or q.shape!=(len(ids),4,4) or not np.isfinite(q).all():
        raise ValueError('Explicit native source IDs and finite camera poses required')
    path=Path(evidence_path);spec=json.loads(path.read_text())
    if spec.get('schema')!='real2sim2real.tracked-pose-evidence/1':raise ValueError('Unknown tracking evidence schema')
    def member(role):
        r=spec[role];p=path.parent/r['path']
        if p.is_symlink() or not p.is_file() or digest(p)!=r['sha256']:raise ValueError('Tracking evidence member changed: '+role)
        return p
    mapping=json.loads(member('mapping').read_text())
    if mapping.get('source_video_sha256')!=source_video_sha256:raise ValueError('Tracking evidence belongs to another source video')
    native=np.asarray(mapping['source_frame_indices'])
    if (native.ndim!=1 or not len(native) or native.dtype.kind not in 'iu' or np.any(native<0)
            or not (np.all(np.diff(native)==1) or np.all(np.diff(native)==-1))):
        raise ValueError('Consecutive monotone native tracking mapping required')
    states={};updates={}
    for role,target in [('states',states),('updates',updates)]:
        for line in member(role).read_text().splitlines():
            row=json.loads(line);i=row.get('frame')
            if type(i) is not int or not 0<=i<len(native) or i in target:raise ValueError('Invalid or duplicate tracker-local frame')
            target[i]=row
    inverse={int(v):i for i,v in enumerate(native)};flags=np.ones(len(ids),dtype=bool);matched=[]
    for j,source in enumerate(ids):
        i=inverse.get(int(source));state=states.get(i);update=updates.get(i)
        if state is None or update is None:continue
        if (state.get('accepted') is not True or update.get('method')!='tracked_update'
                or update.get('recovery') is not False or update.get('original',{}).get('rejected_for')!=[]
                or update.get('selected',{}).get('rejected_for')!=[]):continue
        a=np.asarray(state['selected_pose'],np.float32);b=np.asarray(update['proposed_pose_camera'],np.float32)
        if a.shape!=(4,4) or b.shape!=(4,4) or not np.array_equal(a,b):raise ValueError('Tracker state/update evidence disagree')
        if not np.array_equal(q[j].astype(np.float32),a):continue
        flags[j]=False;matched.append(int(source))
    return flags,dict(schema='real2sim2real.source-matched-validation/1',evidence_sha256=digest(path),
        source_video_sha256=source_video_sha256,tracking_gate_source_frame_ids=matched,
        tracking_source_direction='backward' if len(native)>1 and native[1]<native[0] else 'forward',
        recovery_gate_source_frame_ids=ids[flags].astype(int).tolist(),
        shared_thresholds_changed=False,all_recovery_strength=False,
        explanation='Only unchanged accepted tracked_update poses use the existing tracking gate. All other poses require the existing recovery gate.')
