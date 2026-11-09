"""Recover an exact, hash-bound accepted tracker prefix for a bounded retry."""
import json
from pathlib import Path
import numpy as np
from real2sim2real.artifacts import digest
from real2sim2real.reconstruction.recovery_quality import rigid


def verified_prefix(seed, trace, *, seed_sha256, trace_sha256, source_frames, source_frame_ids=None):
    """Retain measured states only; never carry a pose past a rejected frame.

    The caller owns the source/model/run provenance of these receipts. This
    checks immutable file identity and state-chain consistency, not independent
    image/depth acceptance or complete-reference approval.
    """
    seed,trace=map(Path,(seed,trace))
    if (type(source_frames) is not int or source_frames<2
        or digest(seed)!=seed_sha256 or digest(trace)!=trace_sha256):
        raise ValueError('Verified seed/trace hashes and explicit source coverage required')
    ids=np.arange(source_frames) if source_frame_ids is None else np.asarray(source_frame_ids)
    if (ids.shape!=(source_frames,) or ids.dtype.kind not in 'iu' or np.any(ids<0)
        or not (np.all(np.diff(ids.astype(np.int64))==1) or np.all(np.diff(ids.astype(np.int64))==-1))):
        raise ValueError('Explicit consecutive forward/backward source IDs required')
    initial=json.loads(seed.read_text())
    if initial.get('frame')!=0 or not rigid(np.asarray(initial['pose_cam'])):
        raise ValueError('Prefix must start with an explicit rigid tracker-local-frame-zero seed')
    poses=[np.asarray(initial['pose_cam'],float)];blocked=None
    for line in trace.read_text().splitlines():
        if not line.strip():continue
        row=json.loads(line)
        if type(row.get('accepted')) is not bool:raise ValueError('Explicit accepted state required')
        if blocked is not None:
            if row['accepted']:raise ValueError('Accepted state after a rejected frame cannot form a prefix')
            continue
        if row.get('frame')!=len(poses) or row['frame']>=source_frames:
            raise ValueError('Nonconsecutive or out-of-range tracker source frame')
        previous=np.asarray(row['previous_accepted_pose'],float)
        if not rigid(previous) or not np.allclose(previous,poses[-1],atol=2e-6,rtol=0):
            raise ValueError('Previous accepted pose does not match the retained state chain')
        if not row['accepted']:
            blocked=row['frame'];continue
        value=np.asarray(row['selected_pose'],float)
        if not rigid(value):raise ValueError('Invalid accepted pose')
        poses.append(value)
    return np.asarray(poses),dict(schema='real2sim2real.verified-tracking-prefix/1',
        frames=len(poses),last_source_frame=int(ids[len(poses)-1]),source_frames=source_frames,
        source_frame_ids=ids[:len(poses)].tolist(),last_tracker_frame=len(poses)-1,
        first_rejected_frame=blocked,first_rejected_source_frame=int(ids[blocked]) if blocked is not None else None,
        seed_sha256=seed_sha256,trace_sha256=trace_sha256,
        poses_interpolated=False,source_pose_chain_preserved=True,
        independent_quality_approval=False,full_reference_accepted=False)
