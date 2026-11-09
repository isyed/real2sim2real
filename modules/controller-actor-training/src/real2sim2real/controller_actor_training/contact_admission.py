"""Task-specific contact prerequisites; hand phases are not universal.

This checks controller/contact compatibility only. Source reconstruction,
stability and unassisted physics still require independent validation.
"""
import numpy as np


def assess(hand_contact_active, collision, *, expected_effectors=('hand',), effector_audit=None):
    hand=np.asarray(hand_contact_active)
    if (hand.ndim!=2 or not hand.size or not np.isfinite(hand).all()
            or not np.isin(hand,[0,1]).all() or not expected_effectors
            or len(set(expected_effectors))!=len(expected_effectors)
            or any(not isinstance(x,str) or not x.strip() for x in expected_effectors)):
        raise ValueError('Require explicit binary hand phases and unique expected effectors')
    reasons=[];evidence=[]
    if not collision.get('robot_object_collision'):
        reasons.append('robot_object_collision_disabled')
    for role in expected_effectors:
        if role=='hand':
            if not hand.any():reasons.append('required_hand_phase_missing')
            else:evidence.append(dict(effector='hand',frames=int(np.any(hand,axis=0).sum())))
        else:
            if collision.get('robot_scope')!='all':reasons.append('nonhand_robot_collision_not_enabled')
            rows=(effector_audit or {}).get('effectors',(effector_audit or {}).get('hands',[]))
            row=next((r for r in rows if r.get('side')==role),None)
            if (not effector_audit or effector_audit.get('schema')!='real2sim2real.reference-contact-readiness/1'
                    or not row or type(row.get('frames_within_threshold')) is not int
                    or row['frames_within_threshold']<=0):
                reasons.append('required_effector_proximity_missing:'+role)
            else:evidence.append(dict(effector=role,frames=row['frames_within_threshold']))
    return dict(schema='real2sim2real.contact-admission/1',controller_contact_compatible=not reasons,
        expected_effectors=list(expected_effectors),evidence=evidence,reasons=reasons,
        zero_hand_phases_expected=bool(not hand.any() and 'hand' not in expected_effectors),
        physical_success_established=False,training_ready_established=False,
        reference_phases_modified=False)
