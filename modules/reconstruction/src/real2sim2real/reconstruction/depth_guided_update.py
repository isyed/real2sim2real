"""Opt-in depth-guided translation recovery after an ordinary tracker failure.

No synthetic depth, mesh scaling, interpolated state or relaxed motion limits.
The original RGB proposal and each measured correction remain in the receipt.
"""
import numpy as np
from .conditioned_recovery import select
from .recovery_quality import Limits, rigid
from real2sim2real.reconstruction_repair.depth_translation import bounded_proposals


def finite_ray_search(pose, previous, *, vertices, source_frame, previous_frame, fps,
                      signed_distances_m, measure, max_move_m=.15, limits=Limits()):
    """Opt-in finite hypotheses, with unchanged accepted-state and source gates.

    This does not run automatically after ``correct``. A CPU measurement can
    screen proposals, but independent native validation is still required.
    Selection uses the existing worst-hypothesis IoU ranking, not signed depth
    residual magnitude. Every candidate's absolute-depth evidence is retained.
    """
    from real2sim2real.reconstruction_repair.depth_translation import finite_ray_proposals
    proposals=finite_ray_proposals(vertices,pose,signed_distances_m=signed_distances_m,max_move_m=max_move_m)
    candidates=[dict(name=f'finite_ray_{i}',pose_camera=p['pose_camera'],provenance=p)
                for i,p in enumerate(proposals)]
    result=select(candidates,previous,source_frame=source_frame,previous_frame=previous_frame,
        fps=fps,measure=measure,required_hypotheses=['original'],recovery=True,limits=limits)
    return dict(result,search=dict(kind='explicit_finite_camera_ray',max_move_m=max_move_m,
        requested_distances_m=np.asarray(signed_distances_m,float).tolist(),tested_candidates=len(candidates),
        correction_bound_origin='raw_tracker_proposal',motion_bound_origin='last_accepted_pose',
        quality_limits_changed=False,source_depth_changed=False,rotation_preserved=True))


def correct(pose, previous, *, vertices, source_frame, previous_frame, fps, measure,
            max_move_m=.15, target_residual_m=.03, limits=Limits()):
    if not rigid(pose):raise ValueError('A rigid native tracker proposal is required')
    # Validate the optional configuration even if the original update passes.
    bounded_proposals(vertices,pose,0.,max_move_m=max_move_m,target_residual_m=target_residual_m)
    observed=measure(np.asarray(pose,float))
    if set(observed)!={'original'}:raise ValueError('Use one declared original depth source, not a best-fitting hypothesis')
    context=dict(source_frame=source_frame,previous_frame=previous_frame,fps=fps,
                 required_hypotheses=['original'],limits=limits)
    raw=select([dict(name='normal_tracker',pose_camera=pose,
        provenance=dict(source_frame=source_frame,previous_frame=previous_frame))],previous,
        measure=lambda p:observed,recovery=False,**context)
    result=raw
    proposed=[];reason=None
    if not raw['accepted']:
        g=observed['original']['geometry'];signed=g.get('depth_shift_m')
        if g.get('depth_points',0)<limits.depth_points or signed is None or not np.isfinite(signed):
            reason='insufficient_observed_surface_depth'
        else:
            proposed=bounded_proposals(vertices,pose,signed,max_move_m=max_move_m,target_residual_m=target_residual_m)
            if proposed:
                candidates=[dict(name=f'depth_translation_{i}',pose_camera=r['pose_camera'],provenance=r)
                            for i,r in enumerate(proposed)]
                result=select(candidates,previous,measure=measure,recovery=True,**context)
            else:reason='no_signed_depth_correction'
    # Do not create a recursive receipt when the original passed.
    return dict(result,normal_attempt=raw,depth_guided_translation=dict(
        enabled=True,applied=bool(not raw['accepted'] and result['accepted']),
        raw_measurement=observed,proposals=proposed,skip_reason=reason,
        max_move_m=max_move_m,target_residual_m=target_residual_m,
        quality_limits_changed=False,source_depth_changed=False,mesh_scale_changed=False,
        correction_bound_origin='raw_tracker_proposal',motion_bound_origin='last_accepted_pose'))
