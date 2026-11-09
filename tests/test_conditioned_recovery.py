import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from real2sim2real.reconstruction.conditioned_recovery import select
from real2sim2real.reconstruction_repair.canonical_pose import encode,decode


def pose(x=0):
    p=np.eye(4);p[:3,3]=[x,0,3];return p


def evidence(depth=.02):
    return dict(metrics=dict(iou=.8,precision=.9,recall=.9),geometry=dict(depth_points=500,
        depth_error_m=depth,support_penetration_m=0,behind_camera=False))


def run(measure,p=None):
    p=pose(.02) if p is None else p
    return select([dict(name='fresh',pose_camera=p,provenance='source hashes')],pose(),
        source_frame=128,previous_frame=127,fps=30,measure=measure,required_hypotheses=['original','crop'])


def test_disagreeing_depth_cannot_be_cherry_picked():
    r=run(lambda p:dict(original=evidence(.2),crop=evidence()))
    assert not r['accepted'] and r['candidates'][0]['rejected_by_hypothesis']['original']==['depth']


def test_both_hypotheses_allow_recovery():
    assert run(lambda p:dict(original=evidence(),crop=evidence()))['accepted']


def test_motion_filter_precedes_expensive_measurement():
    r=run(lambda p:pytest.fail('Should not render implausible candidate'),pose(.5))
    assert not r['accepted']


def test_incomplete_evidence_rejected():
    with pytest.raises(ValueError,match='hypothesis'):run(lambda p:dict(original=evidence()))


def test_all_sampler_poses_can_be_kept_without_image_only_preselection():
    c=[dict(name=str(i),pose_camera=pose(.001*i),provenance='hashed') for i in range(52)]
    r=select(c,pose(),source_frame=1,previous_frame=0,fps=30,
        measure=lambda p:dict(original=evidence()),required_hypotheses=['original'])
    assert r['accepted'] and len(r['candidates'])==52


@pytest.mark.parametrize('seed',[0,1,2,3,4])
def test_canonical_conditioning_roundtrip_preserves_metric_vertices(seed):
    rng=np.random.default_rng(seed);p=pose();p[:3,:3]=Rotation.random(random_state=rng).as_matrix()
    Q=Rotation.random(random_state=rng).as_matrix();D=Rotation.random(random_state=rng).as_matrix()
    c=rng.normal(size=3);norm=3.4;v=rng.normal(size=(20,3));a=encode(p,Q,norm,c,D)
    native=((v@Q-c)*norm*a['scale'])@a['rotation']+a['translation'];native=native@D
    assert np.allclose(native,v@p[:3,:3].T+p[:3,3])
    assert np.allclose(decode(**a,axes=Q,normalization=norm,center=c,camera_axes=D),p)


def test_changed_sampler_scale_cannot_silently_resize_metric_mesh():
    a=encode(pose(),np.eye(3),2.,np.zeros(3),np.eye(3));a['scale']*=2
    with pytest.raises(ValueError,match='scale'):decode(**a,axes=np.eye(3),normalization=2.,center=np.zeros(3),camera_axes=np.eye(3))
