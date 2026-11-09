"""Offline recovery admission requiring a successful return to normal tracking.

Only independently qualified current-frame candidates enter the bounded search.
Future observations validate normal tracker updates; they are not interpolated
poses or clean targets. No input candidate/state is modified.
"""
import copy
import numpy as np
from .conditioned_recovery import select
from .recovery_quality import Limits, rigid


def qualify(selection, future_frames, *, fps, step, max_candidates=3, horizon=3,
            limits=Limits()):
    frame=selection['source_frame'];future=list(future_frames)
    if (type(horizon) is not int or not 1<=horizon<=5 or
        type(max_candidates) is not int or not 1<=max_candidates<=5 or
        not np.isfinite(fps) or fps<=0 or not callable(step) or
        future!=list(range(frame+1,frame+1+horizon))):
        raise ValueError('A complete consecutive bounded look-ahead window is required')
    hypotheses=selection['required_hypotheses'];rows=selection['candidates']
    eligible=[i for i,r in enumerate(rows) if r['accepted']]
    ranked=sorted(eligible,key=lambda i:(-min(rows[i]['evidence'][n]['metrics']['iou']
                  for n in hypotheses),i))[:max_candidates]
    trials=[]
    for i in ranked:
        pose=np.asarray(rows[i]['pose_camera'],float).copy();previous_frame=frame
        updates=[];diagnostics=[];scores=[min(rows[i]['evidence'][n]['metrics']['iou'] for n in hypotheses)]
        for f in future:
            # The callback must reset the native tracker to this exact pose.
            proposal,evidence=step(f,pose.copy())
            result=select([dict(name='normal_handback',pose_camera=proposal,
                provenance=dict(candidate_index=i,previous_frame=previous_frame))],pose,
                source_frame=f,previous_frame=previous_frame,fps=fps,
                measure=lambda p:evidence,required_hypotheses=hypotheses,
                recovery=False,limits=limits)
            updates.append(result)
            # Native callbacks may already have rendered evidence before the
            # cheap motion veto. Keep it for diagnosis without overruling that
            # veto or presenting it as accepted evidence.
            p=np.asarray(proposal,float)
            valid=rigid(p)
            diagnostics.append(dict(source_frame=f,previous_frame=previous_frame,
                accepted=result['accepted'],
                translation_m=float(np.linalg.norm(p[:3,3]-pose[:3,3])) if valid else None,
                rotation_deg=float(np.degrees(np.arccos(np.clip(
                    (np.trace(p[:3,:3]@pose[:3,:3].T)-1)/2,-1,1)))) if valid else None,
                measured_evidence=copy.deepcopy(evidence),
                rejected_for=result['candidates'][0]['motion_rejected_for']+
                    sum(result['candidates'][0]['rejected_by_hypothesis'].values(),[])))
            if not result['accepted']:break
            pose=np.asarray(result['pose_camera']);previous_frame=f
            scores.append(min(result['candidates'][0]['evidence'][n]['metrics']['iou'] for n in hypotheses))
        passed=len(updates)==horizon and all(r['accepted'] for r in updates)
        trials.append(dict(candidate_index=i,passed=passed,updates=updates,step_diagnostics=diagnostics,
            minimum_iou=float(min(scores)),mean_iou=float(np.mean(scores))))
    passed=[t for t in trials if t['passed']]
    chosen=max(passed,key=lambda t:(t['minimum_iou'],t['mean_iou'],-t['candidate_index'])) if passed else None
    result=copy.deepcopy(selection)
    result.update(accepted=chosen is not None,selected_index=chosen['candidate_index'] if chosen else None,
        pose_camera=copy.deepcopy(rows[chosen['candidate_index']]['pose_camera']) if chosen else None,
        lookahead=dict(schema='real2sim2real.recovery-lookahead/1',horizon=horizon,
            future_frames=future,max_candidates=max_candidates,qualified_current_frame_candidates=len(eligible),
            current_frame_accepted=selection['accepted'],current_frame_selected_index=selection['selected_index'],
            untested_qualified_candidates=len(eligible)-len(ranked),
            evaluated_future_frames=sorted({u['source_frame'] for t in trials for u in t['updates']}),
            admission_rule='current_frame_and_all_consecutive_future_updates',
            trials=trials,selected_updates=chosen['updates'] if chosen else [],
            passed=chosen is not None,offline_future_observations=True,missing_poses_filled=False))
    return result
