import numpy as np
import pytest
from real2sim2real.reconstruction.hybrid_tracking import HybridConfig,run_sequence,motion_reasons


def pose(x=0):
    p=np.eye(4);p[:3,3]=[x,0,3];return p


def measure(frame,p):
    return dict(metrics=dict(iou=.8,precision=.9,recall=.9),geometry=dict(depth_points=500,
        depth_error_m=.02,support_penetration_m=0,behind_camera=False))


def run(obs,track,provider=None,measurement=measure,config=HybridConfig()):
    n=len(obs);return run_sequence(np.arange(n),np.arange(n)/30,np.array(obs,bool),pose(),
        track=track,measure=measurement,propose=provider,config=config)


def external(p):return [dict(name='external',pose_camera=p,provenance='hashed-source')]


def test_good_tracking_never_calls_expensive_provider():
    def forbidden(*a):raise AssertionError('Recovery unnecessary')
    _,ok,r=run([1,1,1],lambda f,p:pose(.01*f),forbidden)
    assert ok.all() and r['recovery_calls']==0


def test_failed_tracker_replaced_only_with_independent_valid_candidate():
    _,ok,r=run([1,1],lambda f,p:pose(2),lambda f,p:external(pose(.02)))
    assert ok.all() and r['external_accepted_frames']==[1]


@pytest.mark.parametrize('failure',['depth','support','iou','nonrigid','motion'])
def test_bad_external_proposals_never_advance_state(failure):
    p=pose(.02)
    if failure=='nonrigid':p[0,0]=2
    if failure=='motion':p=pose(1)
    def bad_measure(frame,candidate):
        e=measure(frame,candidate)
        if frame:
            if failure=='depth':e['geometry']['depth_error_m']=.5
            if failure=='support':e['geometry']['support_penetration_m']=.1
            if failure=='iou':e['metrics']['iou']=.3
        return e
    a,ok,r=run([1,1],lambda f,p:pose(2),lambda f,old:external(p),measurement=bad_measure)
    assert not ok[1] and np.isnan(a[1]).all() and r['unresolved_observed_frames']==[1]


def test_hidden_frames_are_never_sent_to_models_or_filled():
    called=[]
    def track(f,p):called.append(f);return pose(.03)
    a,ok,r=run([1,0,0,1],track)
    assert called==[3] and ok.tolist()==[1,0,0,1] and np.isnan(a[1:3]).all()
    assert r['observed_coverage_complete'] and not r['full_trajectory_complete']


def test_long_gap_does_not_allow_teleport_or_restart():
    a,ok,r=run([1,0,0,0,1],lambda *a:pytest.fail('No valid motion history'))
    assert r['frames'][-1]['rejected_for']==['observation_gap']


def test_failed_pose_never_used_as_next_track_seed():
    seen=[]
    def track(f,p):seen.append(p.copy());return pose(2 if f==1 else .01)
    _,ok,r=run([1,1,1],track)
    assert np.array_equal(seen[0],seen[1]) and ok.tolist()==[1,0,1]


def test_recovery_budget_and_provenance_are_enforced():
    calls=[]
    def provider(f,p):calls.append(f);return []
    _,_,r=run([1,1,1,1],lambda f,p:pose(2),provider,config=HybridConfig(max_recovery_calls=1))
    assert calls==[1] and r['frames'][2]['recovery_budget_exhausted']
    with pytest.raises(ValueError,match='provenance'):
        run([1,1],lambda f,p:pose(2),lambda f,p:[dict(pose_camera=pose())])


def test_time_rate_scales_only_with_actual_gap():
    assert not motion_reasons(pose(.12),pose(),elapsed_seconds=2/30)
    assert 'translation_step' in motion_reasons(pose(.12),pose(),elapsed_seconds=1/30)
    assert 'observation_gap' in motion_reasons(pose(.12),pose(),elapsed_seconds=1.)


def test_recovery_uses_stricter_existing_image_threshold():
    def measurement(f,p):
        e=measure(f,p)
        if f:e['metrics']['iou']=.57
        return e
    _,ok,_=run([1,1],lambda f,p:pose(.01),measurement=measurement)
    assert ok.all()
    _,ok,r=run([1,1],lambda f,p:pose(2),lambda f,p:external(pose(.01)),measurement=measurement)
    assert not ok[1] and 'iou' in r['frames'][1]['external_proposals'][0]['rejected_for']
