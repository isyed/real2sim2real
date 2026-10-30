"""Decisions for tracking-first repair; calibration and tracking stay separate.

A consistent metric hypothesis may resize a *new* tracking input. Old poses are
never rescaled into accepted motion. Rejected calibration retains the original
mesh and permits only a bounded registration diagnosis; normal fine repair and
reference selection remain blocked until metric evidence is resolved.
"""
import numpy as np


def plan(calibration, *, retained_size_ratio=1.10):
    if retained_size_ratio<=1:raise ValueError('Retained-size ratio must exceed one')
    consistent=calibration.get('candidate_ready') is True
    available=calibration.get('scale_multiplier') is not None
    if consistent and not available:raise ValueError('Passing calibration must provide a scale')
    factor=float(calibration['scale_multiplier']) if available else 1.
    if not np.isfinite(factor) or factor<=0:raise ValueError('Positive finite calibration scale required')
    resize=consistent and not 1/retained_size_ratio<=factor<=retained_size_ratio
    return dict(schema='real2sim2real.tracking-repair-plan/1',
        calibration_consistent=consistent,calibration_scale_available=available,metric_reasons=calibration.get('reasons',[]),
        mesh_scale_multiplier=factor if resize else 1.,resize_hypothesis_retained=resize,
        old_pose_sequence_reusable=False,fresh_registration_required=True,
        full_tracking_permitted=consistent,fine_repair_permitted=consistent and not resize,
        tracking_only_diagnostic=not consistent,
        next_action='resize_and_register_then_retrack' if resize else 'register_then_retrack' if consistent else 'bounded_registration_diagnosis_then_resolve_metric_evidence',
        selected_reference_changed=False,actor_training_approved=False)


def assess_replacement(parent_iou, candidate_iou, *, expected_frames, accepted_state_frames,
                       continuity_passed, native_views_passed, metric_passed,
                       minimum_mean_gain=.002,max_good_frame_drop=.02,good_iou=.60):
    """Select only complete, measured tracking improvements under unchanged gates."""
    old,new=np.asarray(parent_iou,float),np.asarray(candidate_iou,float)
    reasons=[]
    if old.shape!=(expected_frames,) or new.shape!=(expected_frames,):
        raise ValueError('Full parent/candidate image scores must match the source timeline')
    if not np.isfinite(old).all() or not np.isfinite(new).all():raise ValueError('Finite image scores required')
    if list(accepted_state_frames)!=list(range(1,expected_frames)):reasons.append('incomplete_accepted_state_trace')
    if not continuity_passed:reasons.append('motion_continuity')
    if not native_views_passed:reasons.append('native_image_depth_support')
    if not metric_passed:reasons.append('current_mesh_metric_evidence')
    regressions=np.flatnonzero((old>=good_iou)&(new<old-max_good_frame_drop)).tolist()
    if regressions:reasons.append('previous_good_images_regressed')
    gain=float(new.mean()-old.mean())
    if gain<minimum_mean_gain:reasons.append('no_mean_image_improvement')
    return dict(accepted=not reasons,reasons=reasons,mean_iou_before=float(old.mean()),mean_iou_after=float(new.mean()),
        mean_iou_gain=gain,previous_good_regression_frames=regressions,expected_frames=expected_frames,
        source_timeline_preserved=True,actor_training_approved=False,physics_validated=False)
