"""Lazy external recovery proposals for an independently checked pose tracker.

No model dependency or episode-specific assumptions. A failed observation never
advances the last accepted state. Missing poses remain NaN with an explicit flag;
this diagnostic output must not be published as a complete robot reference.
"""
from dataclasses import dataclass,replace,asdict
import numpy as np
from .recovery_quality import Limits,rigid,rejection_reasons


@dataclass(frozen=True)
class HybridConfig:
    nominal_fps: float = 30.
    max_gap_frames: int = 3
    max_recovery_calls: int = 6
    max_proposals_per_call: int = 2

    def validate(self):
        if (not np.isfinite(self.nominal_fps) or self.nominal_fps<=0 or
            any(type(x) is not int or x<1 for x in
                (self.max_gap_frames,self.max_recovery_calls,self.max_proposals_per_call)) or
            self.max_gap_frames>10 or self.max_proposals_per_call>32):
            raise ValueError('Explicit finite frame rate and bounded integer search/gap limits required')


def motion_reasons(pose,previous,*,elapsed_seconds,config=HybridConfig(),limits=Limits()):
    """Convert existing per-frame displacement limits to a bounded time rate.

    Gaps do not get unlimited displacement: only max_gap_frames are admissible.
    This checks pose origins in the common fixed camera; moving cameras must be
    transformed to a common frame before use.
    """
    config.validate()
    if not rigid(pose) or not rigid(previous):return ['nonrigid']
    if not np.isfinite(elapsed_seconds) or elapsed_seconds<=0:return ['invalid_time']
    steps=elapsed_seconds*config.nominal_fps
    if steps>config.max_gap_frames+1e-6:return ['observation_gap']
    from scipy.spatial.transform import Rotation
    reasons=[]
    if np.linalg.norm(np.asarray(pose)[:3,3]-np.asarray(previous)[:3,3])>limits.step_translation_m*steps+1e-7:
        reasons.append('translation_step')
    angle=np.degrees(Rotation.from_matrix(np.asarray(pose)[:3,:3]@np.asarray(previous)[:3,:3].T).magnitude())
    if angle>limits.step_rotation_deg*steps+1e-6:reasons.append('rotation_step')
    return reasons


def assess_candidate(pose,previous,evidence,*,elapsed_seconds,recovery,
                     config=HybridConfig(),limits=Limits()):
    """Image, depth and support checks remain unchanged; add explicit time join."""
    motion=motion_reasons(pose,previous,elapsed_seconds=elapsed_seconds,config=config,limits=limits)
    if not rigid(pose) or not rigid(previous):return ['nonrigid']
    steps=elapsed_seconds*config.nominal_fps if np.isfinite(elapsed_seconds) and elapsed_seconds>0 else 1.
    scaled=replace(limits,step_translation_m=limits.step_translation_m*steps+1e-7,
        step_rotation_deg=limits.step_rotation_deg*steps+1e-6)
    reasons=rejection_reasons(pose,previous,evidence.get('metrics',{}),evidence.get('geometry',{}),
        recovery=recovery,limits=scaled)
    return list(dict.fromkeys(motion+reasons))


def run_sequence(frame_ids,timestamps,observed,initial_pose,*,track,measure,propose=None,
                 config=HybridConfig(),limits=Limits(),record=None):
    """Run normal tracking first, calling a proposal provider only on failure.

    track(frame_id,last_pose) must reset its internal state to last_pose before
    each update. measure(frame_id,pose) independently returns metrics/geometry.
    propose(frame_id,last_pose) returns bounded dicts with pose_camera, name and
    provenance. Neither its source score nor acceptance flag is trusted.
    """
    config.validate();ids=np.asarray(frame_ids);ts=np.asarray(timestamps,float);obs=np.asarray(observed)
    if (ids.ndim!=1 or len(ids)<2 or ids.dtype.kind not in 'iu' or np.any(ids<0)
        or np.any(np.diff(ids)!=1) or ts.shape!=ids.shape or not np.isfinite(ts).all()
        or np.any(np.diff(ts)<=0) or obs.shape!=ids.shape or obs.dtype!=bool or not obs[0]
        or not rigid(initial_pose) or not callable(track) or not callable(measure)
        or (propose is not None and not callable(propose))):
        raise ValueError('Consecutive source IDs, ordered timestamps, observations and rigid anchor required')
    out=np.full((len(ids),4,4),np.nan);accepted=np.zeros(len(ids),bool);events=[];calls=0
    initial=np.asarray(initial_pose,float).copy();evidence=measure(int(ids[0]),initial)
    reasons=rejection_reasons(initial,initial,evidence.get('metrics',{}),evidence.get('geometry',{}),recovery=True,limits=limits)
    if reasons:raise ValueError('Initial anchor failed independent checks: '+','.join(reasons))
    out[0]=initial;accepted[0]=True;last=0
    row=dict(frame=int(ids[0]),status='anchor',accepted=True,pose_camera=initial.tolist(),evidence=evidence)
    events.append(row)
    if record:record(row)
    for i in range(1,len(ids)):
        frame=int(ids[i]);previous=out[last].copy();dt=float(ts[i]-ts[last])
        row=dict(frame=frame,accepted=False,previous_accepted_frame=int(ids[last]),elapsed_seconds=dt,
            observed=bool(obs[i]),external_proposals=[])
        selected=None
        if not obs[i]:row.update(status='unobserved',rejected_for=['missing_observation'])
        elif dt*config.nominal_fps>config.max_gap_frames+1e-6:
            row.update(status='unresolved',rejected_for=['observation_gap'])
        else:
            current=np.asarray(track(frame,previous),float)
            checked=measure(frame,current) if rigid(current) else {}
            rejected=assess_candidate(current,previous,checked,elapsed_seconds=dt,recovery=False,config=config,limits=limits)
            row['tracker']=dict(pose_camera=current.tolist(),evidence=checked,rejected_for=rejected)
            if not rejected:selected=current;row['status']='tracked'
            else:
                row.update(status='unresolved',rejected_for=rejected)
                if propose is not None and calls<config.max_recovery_calls:
                    calls+=1;pool=list(propose(frame,previous));row['recovery_call']=calls
                    if len(pool)>config.max_proposals_per_call:raise ValueError('External proposal budget exceeded')
                    eligible=[]
                    for item in pool:
                        if not item.get('name') or not item.get('provenance'):raise ValueError('Proposal provenance required')
                        p=np.asarray(item['pose_camera'],float)
                        bad=motion_reasons(p,previous,elapsed_seconds=dt,config=config,limits=limits)
                        ev={} if bad else measure(frame,p)
                        if not bad:bad=assess_candidate(p,previous,ev,elapsed_seconds=dt,recovery=True,config=config,limits=limits)
                        receipt=dict(name=item['name'],provenance=item['provenance'],pose_camera=p.tolist(),
                            evidence=ev,rejected_for=bad,independently_measured=bool(ev))
                        row['external_proposals'].append(receipt)
                        if not bad:eligible.append((float(ev['metrics']['iou']),p,receipt))
                    if eligible:
                        _,selected,chosen=max(eligible,key=lambda x:x[0]);row.update(status='external_recovery',selected_proposal=chosen['name'])
                elif propose is not None:row['recovery_budget_exhausted']=True
        if selected is not None:
            out[i]=selected;accepted[i]=True;last=i
            row.update(accepted=True,pose_camera=selected.tolist(),rejected_for=[])
        events.append(row)
        if record:record(row)
    result=dict(schema='real2sim2real.hybrid-tracking/1',config=asdict(config),limits=asdict(limits),
        frames=events,recovery_calls=calls,external_accepted_frames=[r['frame'] for r in events if r['status']=='external_recovery'],
        unresolved_observed_frames=ids[obs&~accepted].tolist(),unobserved_frames=ids[~obs].tolist(),
        observed_coverage_complete=bool(accepted[obs].all()),full_trajectory_complete=bool(accepted.all()),
        missing_poses_filled=False,depth_is_prediction=True,physics_validated=False,reference_approved=False)
    return out,accepted,result
