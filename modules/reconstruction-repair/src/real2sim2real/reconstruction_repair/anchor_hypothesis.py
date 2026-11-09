"""Construct an explicitly unaccepted pose hypothesis between measured anchors."""
import numpy as np
from scipy.spatial.transform import Rotation,Slerp
from real2sim2real.reconstruction.recovery_quality import rigid


def initialize(frame_ids,anchor_ids,anchor_poses,*,fps):
    ids=np.asarray(frame_ids);anchors=np.asarray(anchor_ids);poses=np.asarray(anchor_poses,float)
    if (ids.ndim!=1 or ids.dtype.kind not in 'iu' or len(ids)<5 or np.any(np.diff(ids)!=1)
        or anchors.ndim!=1 or anchors.dtype.kind not in 'iu' or len(anchors)<2
        or np.any(np.diff(anchors)<=0) or anchors[0]!=ids[0] or anchors[-1]!=ids[-1]
        or not np.isin(anchors,ids).all() or poses.shape!=(len(anchors),4,4)
        or not all(rigid(p) for p in poses) or not np.isfinite(fps) or fps<=0):
        raise ValueError('Consecutive source frames bracketed by rigid ordered anchors and explicit FPS required')
    result=np.tile(np.eye(4),(len(ids),1,1));times=ids/float(fps);at=anchors/float(fps)
    result[:,:3,3]=np.stack([np.interp(times,at,poses[:,j,3]) for j in range(3)],axis=1)
    result[:,:3,:3]=Slerp(at,Rotation.from_matrix(poses[:,:3,:3]))(times).as_matrix()
    locked=np.isin(ids,anchors);result[locked]=poses
    return result,dict(schema='real2sim2real.anchor-hypothesis/1',source_frame_ids=ids.tolist(),
        anchor_frame_ids=anchors.tolist(),hypothesis_only=True,accepted_reference=False,
        interpolated_frames_verified=False,temporal_truth_known=False)
