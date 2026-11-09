"""Version-compatible observed-pose seeds with explicit reuse provenance."""
import re
import numpy as np
from .recovery_quality import rigid

SOURCE='observed_rgb_silhouette_depth_fit'


def validate(value):
    """Check the exporter's seed contract before allocating models or a GPU.

    This validates format and declared provenance, not image/depth quality.
    Reused poses still require caller-owned state/measurement verification.
    """
    if (not isinstance(value,dict) or value.get('source')!=SOURCE
        or type(value.get('frame')) is not int or value['frame']!=0
        or value.get('clean_trajectory_used',False) is not False):
        raise ValueError('Require the canonical observed-seed source tag, local frame zero and no clean trajectory')
    pose=np.asarray(value.get('pose_cam'),float);camera=np.asarray(value.get('K'),float)
    if (not rigid(pose) or camera.shape!=(3,3) or not np.isfinite(camera).all()
        or min(camera[0,0],camera[1,1])<=0 or not np.allclose(camera[2],[0,0,1])):
        raise ValueError('Rigid camera pose and finite positive calibrated intrinsics required')
    return pose.copy(),camera.copy()


def make(pose,camera,*,native_source_frame,method,evidence_hashes):
    """Keep the compatibility tag fixed; describe reuse in separate fields."""
    if (type(native_source_frame) is not int or native_source_frame<0
        or method not in ('verified_tracking_prefix','gpu_revalidated_local_recovery')
        or not isinstance(evidence_hashes,dict) or not evidence_hashes
        or any(not isinstance(k,str) or not k or not isinstance(v,str)
               or re.fullmatch(r'[0-9a-f]{64}',v) is None for k,v in evidence_hashes.items())):
        raise ValueError('Explicit source ID, supported observed reuse method and SHA-256 evidence required')
    value=dict(source=SOURCE,frame=0,pose_cam=np.asarray(pose,float).tolist(),K=np.asarray(camera,float).tolist(),
        native_source_frame=native_source_frame,method=method,evidence_sha256=dict(evidence_hashes),
        clean_trajectory_used=False,format_validation_is_not_quality_approval=True)
    validate(value);return value


def failure_kind(returncode,failure=None):
    """Do not interpret a startup/configuration exception as a tracking result."""
    if returncode==0:return 'completed'
    if returncode==86:return 'guard_stop'
    message=(failure or {}).get('error','')
    if message.startswith('No quality-valid recovery at '):return 'quality_rejection'
    if message=='Recovery-frame budget exhausted':return 'work_budget'
    return 'runtime_failure'
