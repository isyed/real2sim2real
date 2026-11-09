import copy
import numpy as np
import pytest
from real2sim2real.reconstruction.conditioned_recovery import select
from real2sim2real.reconstruction.recovery_lookahead import qualify


def pose(x=0):
    p=np.eye(4);p[:3,3]=[x,0,3];return p


def evidence(iou=.8,depth=.02):
    return {'original':dict(metrics=dict(iou=iou,precision=.9,recall=.9),
        geometry=dict(depth_points=500,depth_error_m=depth,support_penetration_m=0,behind_camera=False))}


def current():
    return select([dict(name=str(i),pose_camera=pose(x),provenance='native checked')
        for i,x in enumerate([.01,.02,.03])],pose(),source_frame=10,previous_frame=9,fps=30,
        measure=lambda p:evidence(.95 if p[0,3]==.01 else .8),required_hypotheses=['original'])


def test_best_single_frame_candidate_can_lose_to_stable_handback():
    r=current();original=copy.deepcopy(r)
    def step(f,p):return p,evidence(depth=.2 if p[0,3]==.01 else .02)
    out=qualify(r,[11,12,13],fps=30,step=step)
    assert out['accepted'] and out['selected_index']==1
    assert not out['lookahead']['trials'][0]['passed']
    assert len(out['lookahead']['selected_updates'])==3 and r==original


def test_no_safe_handback_does_not_commit_a_pose():
    r=qualify(current(),[11,12,13],fps=30,step=lambda f,p:(p,evidence(depth=.2)))
    assert not r['accepted'] and r['pose_camera'] is None


def test_failed_future_motion_is_rejected_even_with_good_overlap():
    r=qualify(current(),[11,12,13],fps=30,step=lambda f,p:(pose(.5),evidence()))
    assert not r['accepted']


@pytest.mark.parametrize('frames',[[11,13],[11,12],[12,13,14]])
def test_cannot_accept_truncated_or_gapped_window(frames):
    with pytest.raises(ValueError):qualify(current(),frames,fps=30,step=lambda f,p:(p,evidence()))


def test_work_bound_and_no_sampling_for_rejected_candidates():
    calls=[];r=current();r['candidates'][0]['accepted']=False
    out=qualify(r,[11,12,13],fps=30,step=lambda f,p:(calls.append(f) or p,evidence()),max_candidates=1)
    assert out['accepted'] and calls==[11,12,13]


def test_missing_depth_evidence_cannot_pass():
    with pytest.raises(ValueError,match='hypothesis'):
        qualify(current(),[11,12,13],fps=30,step=lambda f,p:(p,{}))


def test_motion_rejection_preserves_measured_evidence_without_admitting_pose():
    measured=evidence(depth=.2)
    out=qualify(current(),[11,12,13],fps=30,step=lambda f,p:(pose(.5),measured))
    trial=out['lookahead']['trials'][0];d=trial['step_diagnostics'][0]
    assert not out['accepted'] and out['pose_camera'] is None
    assert d['translation_m']==pytest.approx(.49)
    assert d['rotation_deg']==pytest.approx(0)
    assert d['measured_evidence']==measured
    assert d['rejected_for']==['translation_step']
    assert trial['updates'][0]['candidates'][0]['evidence']=={}
    measured['original']['geometry']['depth_error_m']=0
    assert d['measured_evidence']['original']['geometry']['depth_error_m']==.2


def test_report_distinguishes_single_frame_success_and_bounded_future_failure():
    out=qualify(current(),[11,12,13],fps=30,step=lambda f,p:(pose(.5),evidence()),max_candidates=1)
    x=out['lookahead']
    assert x['current_frame_accepted'] and not x['passed']
    assert x['current_frame_selected_index']==0 and out['selected_index'] is None
    assert x['future_frames']==[11,12,13] and x['evaluated_future_frames']==[11]
    assert x['untested_qualified_candidates']==2


def test_future_updates_use_each_new_accepted_pose_without_changing_input():
    source=current();original=copy.deepcopy(source);seen=[]
    def step(f,p):
        seen.append(p[0,3]);p[0,3]+=.01
        return p,evidence()
    out=qualify(source,[11,12,13],fps=30,step=step,max_candidates=1)
    assert out['accepted'] and source==original
    assert seen==pytest.approx([.01,.02,.03])
    assert [u['previous_frame'] for u in out['lookahead']['selected_updates']]==[10,11,12]
    assert out['lookahead']['evaluated_future_frames']==[11,12,13]
