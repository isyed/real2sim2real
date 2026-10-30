import numpy as np
import pytest
from real2sim2real.reconstruction.initial_registration import choose_seed


def evidence(iou=.8, depth=.02, penetration=0.):
    return dict(iou=iou, precision=.9, recall=.9), dict(depth_points=200,
        depth_error_m=depth, support_penetration_m=penetration, behind_camera=False)


def test_initial_pose_must_pass_depth_and_support_not_only_image_score():
    poses=np.tile(np.eye(4),(3,1,1));pairs=[evidence(.99,.23),evidence(.95,.02,.035),evidence(.75)]
    chosen,reports=choose_seed(poses,[p[0] for p in pairs],[p[1] for p in pairs])
    assert chosen==2
    assert reports[0]['rejected_for']==['depth']
    assert reports[1]['rejected_for']==['support']


def test_no_eligible_first_pose_returns_no_seed():
    m,g=evidence(.59);chosen,_=choose_seed(np.eye(4)[None],[m],[g])
    assert chosen is None


def test_initial_registration_has_no_invented_previous_pose():
    pose=np.eye(4);pose[:3,3]=[20.,10.,30.];m,g=evidence()
    chosen,reports=choose_seed(pose[None],[m],[g])
    assert chosen==0 and not reports[0]['rejected_for']


def test_initial_candidate_evidence_lengths_are_checked():
    with pytest.raises(ValueError,match='matching lengths'):
        choose_seed(np.eye(4)[None],[],[])


def test_probe_restores_upstream_hooks_after_an_execution_error(tmp_path):
    from types import SimpleNamespace
    from real2sim2real.reconstruction.initial_registration import probe
    original_iterator=lambda *a,**k: iter(())
    def fail(*args,**kwargs):raise RuntimeError('model startup failed')
    upstream=SimpleNamespace(FoundationPose=object,_foundationpose_frame_iterator=original_iterator,
        run_foundationpose_mhr_export=fail,FOUNDATIONPOSE_SEQUENCE_MODE_REGISTER_FIRST_THEN_TRACK='tracking')
    support=dict(ground_truth_used=False,coordinate_frame='camera',evidence='own depth')
    with pytest.raises(RuntimeError,match='model startup failed'):
        probe(upstream,tmp_path/'export',tmp_path/'depth',support,tmp_path/'out')
    assert upstream.FoundationPose is object
    assert upstream._foundationpose_frame_iterator is original_iterator
