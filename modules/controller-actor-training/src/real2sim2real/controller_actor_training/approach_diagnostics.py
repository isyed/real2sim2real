"""Measure short approach trials without rewarding contact caused by a fall.

The pelvis threshold is a declared diagnostic, not a task-success classifier.
Contact is useful to the pilot only before its first low-pelvis event and during
the corresponding hand's reference phase. This module never changes rewards.
"""
import math


def summarize(body_q, reference_q, contacts, hand_ids, reference_contact, *,
              pelvis_body, palm_bodies, fps, pelvis_threshold_m=.35, early_seconds=4.):
    import numpy as np
    q=np.asarray(body_q);target=np.asarray(reference_q)
    counts=np.asarray(contacts);slots=np.asarray(hand_ids);phase=np.asarray(reference_contact,bool)
    if (q.ndim!=3 or q.shape!=target.shape or q.shape[-1]!=7 or len(q)<2
            or counts.ndim!=2 or len(counts)!=len(q) or slots.shape!=(counts.shape[1],)
            or phase.shape!=(len(q),2) or len(palm_bodies)!=2
            or not all(np.isfinite(x).all() for x in [q,target,counts])
            or not math.isfinite(fps) or fps<=0 or not math.isfinite(pelvis_threshold_m)
            or not math.isfinite(early_seconds) or early_seconds<=0):
        raise ValueError('Complete finite matched states, two-hand phases and explicit clock required')
    if not np.isin(slots,[0,1]).all():raise ValueError('Unknown contact hand')
    n=len(q);duration=(n-1)/fps;low=np.flatnonzero(q[:,pelvis_body,2]<pelvis_threshold_m)
    first=int(low[0]) if len(low) else n
    before=np.arange(n)<first
    measured=np.column_stack([(counts[:,slots==side]>0).any(axis=1) for side in (0,1)])
    useful=measured & phase & before[:,None]
    def longest(flags):
        edges=np.diff(np.r_[False,flags,False].astype(int))
        lengths=np.flatnonzero(edges==-1)-np.flatnonzero(edges==1)
        return min(duration,float(lengths.max(initial=0)/fps))
    end=min(n,int(round(early_seconds*fps)))
    errors=np.sum((q[:end,palm_bodies,:3]-target[:end,palm_bodies,:3])**2,axis=-1)
    upright_errors=errors[before[:end]]
    return dict(survival_s=min(duration,first/fps),window_s=duration,
        first_low_pelvis_frame=first if len(low) else None,pelvis_threshold_m=pelvis_threshold_m,
        longest_hand_contact_s=max(longest(measured[:,side]) for side in (0,1)),
        upright_intended_contact_s=max(longest(useful[:,side]) for side in (0,1)),
        upright_intended_contact_frames_lr=useful.sum(axis=0).tolist(),
        palm_rmse_first4s_m=float(np.sqrt(errors.mean())),
        upright_palm_rmse_first4s_m=float(np.sqrt(upright_errors.mean())) if len(upright_errors) else None,
        no_low_pelvis_event=not len(low),contact_is_not_grasp_success=True)


def continuation(baseline, current, *, frozen_auc, auc_tolerance=.03):
    """Require physical improvement over a declared comparison controller."""
    fields=['survival_s','min_survival_s','upright_intended_contact_s','upright_palm_rmse_first4s_m','auc']
    if not current.get('valid') or any(not isinstance(row.get(key),(int,float)) or
            not math.isfinite(row[key]) for row in (baseline,current) for key in fields):
        return dict(continue_to_10m=False,reason='Invalid or incomplete evaluation')
    stable=(current['survival_s']>=baseline['survival_s']-.2 and
            current['min_survival_s']>=baseline['min_survival_s']-.2)
    balance=current['survival_s']>=baseline['survival_s']+.5
    contact=current['upright_intended_contact_s']>=max(.15,baseline['upright_intended_contact_s']+.1)
    reach=current['upright_palm_rmse_first4s_m']<=baseline['upright_palm_rmse_first4s_m']*.85
    tracking=current['auc']>=frozen_auc-auc_tolerance
    return dict(continue_to_10m=bool(stable and tracking and (balance or contact or reach)),
        stable=stable,balance_improved=balance,upright_contact_improved=contact,
        upright_palm_tracking_improved=reach,own_auc_guard_passed=tracking,
        contact_after_low_pelvis_excluded=True)
