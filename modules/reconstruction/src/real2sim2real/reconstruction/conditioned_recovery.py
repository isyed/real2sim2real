"""Bounded recovery selection with explicit competing depth hypotheses.

Source disagreement does not license choosing whichever depth makes a pose pass.
Each required hypothesis must pass independently. Inference conditioning is the
last accepted state, never a rejected tracker update or another sampler chain.
"""
import numpy as np
from .hybrid_tracking import HybridConfig,motion_reasons,assess_candidate
from .recovery_quality import Limits,rigid


def select(candidates,previous,*,source_frame,previous_frame,fps,measure,required_hypotheses,
           recovery=True,max_candidates=64,limits=Limits()):
    names=list(required_hypotheses)
    if (type(source_frame) is not int or type(previous_frame) is not int or previous_frame<0
        or source_frame<=previous_frame or not rigid(previous) or not np.isfinite(fps) or fps<=0
        or not names or len(names)!=len(set(names)) or any(not isinstance(x,str) or not x for x in names)
        or type(max_candidates) is not int or not 1<=max_candidates<=64 or len(candidates)>max_candidates):
        raise ValueError('Accepted rigid history, explicit source times and bounded named evidence required')
    config=HybridConfig(nominal_fps=float(fps));elapsed=(source_frame-previous_frame)/fps
    rows=[];eligible=[]
    for item in candidates:
        if not item.get('name') or not item.get('provenance'):raise ValueError('Every candidate needs provenance')
        pose=np.asarray(item['pose_camera'],float)
        motion=motion_reasons(pose,previous,elapsed_seconds=elapsed,config=config,limits=limits)
        evidence={} if motion else measure(pose);failures={}
        if not motion:
            if set(evidence)!=set(names):raise ValueError('Missing or unexpected independent evidence hypothesis')
            failures={name:assess_candidate(pose,previous,evidence[name],elapsed_seconds=elapsed,
                recovery=recovery,config=config,limits=limits) for name in names}
        accepted=not motion and not any(failures.values())
        row=dict(name=item['name'],provenance=item['provenance'],pose_camera=pose.tolist(),
                 motion_rejected_for=motion,evidence=evidence,rejected_by_hypothesis=failures,accepted=accepted)
        rows.append(row)
        if accepted:eligible.append((min(evidence[n]['metrics']['iou'] for n in names),len(rows)-1))
    chosen=max(eligible,key=lambda x:x[0])[1] if eligible else None
    return dict(source_frame=source_frame,previous_frame=previous_frame,elapsed_seconds=elapsed,
        previous_pose_camera=np.asarray(previous).tolist(),required_hypotheses=names,candidates=rows,
        accepted=chosen is not None,selected_index=chosen,
        pose_camera=rows[chosen]['pose_camera'] if chosen is not None else None,
        missing_pose_filled=False,reference_approved=False)
