"""Retain admissible original observations after a temporal fit regresses them.

This generates a hybrid proposal, then rechecks the ENTIRE native window and
all joins. A good silhouette alone cannot qualify a pose with invalid depth.
"""
from dataclasses import asdict
import numpy as np
from real2sim2real.reconstruction.recovery_quality import rigid
from .temporal_pose import TemporalPoseConfig,validate,measurements,acceptance


def preserve(data,candidate,*,config=TemporalPoseConfig(benefit_mode='observation')):
    config.validate();validate(data);original=np.asarray(data['poses_camera']);candidate=np.asarray(candidate)
    if candidate.shape!=original.shape or not all(rigid(p) for p in candidate):raise ValueError('Matching rigid candidate timeline required')
    before=measurements(data,original,config);proposed=measurements(data,candidate,config)
    hybrid=candidate.copy();restored=[]
    for i,(a,b) in enumerate(zip(before,proposed)):
        valid=(a['iou']>=config.min_iou and a['precision']>=config.min_precision and a['recall']>=config.min_recall
            and a['depth_points']>=config.min_depth_points and a['depth_error_m'] is not None
            and a['depth_error_m']<=config.max_depth_error_m and a['support_penetration_m']<=config.max_support_penetration_m
            and not a['behind_camera'])
        if valid and data['fit'][i] and b['iou']<a['iou']-config.max_frame_iou_drop and not data['locked'][i]:
            hybrid[i]=original[i];restored.append(int(data['frame_ids'][i]))
    after=measurements(data,hybrid,config)
    decision=acceptance(data,before,after,original,hybrid,config)
    return hybrid,dict(schema='real2sim2real.observation-preservation/1',before=before,proposed=proposed,after=after,
        decision=decision,restored_source_frame_ids=restored,config=asdict(config),
        action='restore_only_original_poses_passing_all_native_observation_gates_then_recheck_window',
        restoration_uses_fit_frames_only=True,heldout_frames_used_for_restoration=False,
        thresholds_relaxed=False,all_window_and_motion_gates_rechecked=True,
        cpu_proposal_only=True,requires_gpu_validation=True,complete_reference_approved=False,physics_validated=False)
