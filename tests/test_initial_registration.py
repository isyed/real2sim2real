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


@pytest.mark.parametrize('mode,rgb_only',[('rgb',True),('depth',False)])
def test_registration_proposal_mode_is_explicit_and_keeps_independent_gates(tmp_path,monkeypatch,mode,rgb_only):
    from types import SimpleNamespace
    from real2sim2real.reconstruction import initial_registration as module
    recorded=[]
    class Base:
        mesh_ori=SimpleNamespace(vertices=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]]))
        poses=np.eye(4)[None]
        def get_tf_to_centered_mesh(self):return np.eye(4)
        def register(self,*args,**kwargs):recorded.append(kwargs)
    K=np.eye(3);mask=np.ones((12,12),bool)
    def iterator(*a,**k):yield (0,np.zeros((12,12,3)),np.ones((12,12)),~mask,mask,K)
    upstream=SimpleNamespace(FoundationPose=Base,_foundationpose_frame_iterator=iterator,
        _tensor_to_numpy=np.asarray,_render_candidate_masks=lambda *a:mask[None],
        measure_visible_silhouettes=lambda *a:None,
        _selected_metric_values=lambda *a:dict(iou=.8,precision=.9,recall=.9),
        FOUNDATIONPOSE_SEQUENCE_MODE_REGISTER_FIRST_THEN_TRACK='tracking')
    def execute(*args,**kwargs):
        for _ in upstream._foundationpose_frame_iterator():upstream.FoundationPose().register()
    upstream.run_foundationpose_mhr_export=execute
    monkeypatch.setattr(module,'consistency',lambda *a:dict(depth_points=144,depth_error_m=.01,support_penetration_m=0.,behind_camera=False))
    report=module.probe(upstream,tmp_path/'export',tmp_path/'depth',
        dict(ground_truth_used=False,coordinate_frame='camera',evidence='synthetic source evidence'),
        tmp_path/'output',proposal_mode=mode)
    assert recorded[0]['rgb_only'] is rgb_only
    assert not recorded[0]['both_depth_and_rgb']
    assert report['accepted'] and not report['thresholds_relaxed']
    assert report['proposal_source']=='fresh_'+mode+'_registration'
    assert upstream.FoundationPose is Base


def test_later_visible_frame_is_indexed_without_changing_gates(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from real2sim2real.reconstruction import initial_registration as module
    class Base:
        mesh_ori=SimpleNamespace(vertices=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]]))
        poses=np.eye(4)[None]
        def get_tf_to_centered_mesh(self):return np.eye(4)
        def register(self,*args,**kwargs):pass
    K=np.eye(3);mask=np.ones((12,12),bool);called={}
    def iterator(*a,**k):yield ('000300',np.zeros((12,12,3)),np.ones((12,12)),~mask,mask,K)
    upstream=SimpleNamespace(FoundationPose=Base,_foundationpose_frame_iterator=iterator,
        _tensor_to_numpy=np.asarray,_render_candidate_masks=lambda *a:mask[None],
        measure_visible_silhouettes=lambda *a:None,
        _selected_metric_values=lambda *a:dict(iou=.8,precision=.9,recall=.9),
        FOUNDATIONPOSE_SEQUENCE_MODE_REGISTER_FIRST_THEN_TRACK='tracking')
    def execute(*args,**kwargs):
        called.update(kwargs)
        for _ in upstream._foundationpose_frame_iterator():upstream.FoundationPose().register()
    upstream.run_foundationpose_mhr_export=execute
    monkeypatch.setattr(module,'consistency',lambda *a:dict(depth_points=144,depth_error_m=.01,support_penetration_m=0.,behind_camera=False))
    report=module.probe(upstream,tmp_path/'export',tmp_path/'depth',
        dict(ground_truth_used=False,coordinate_frame='camera',evidence='synthetic source evidence'),
        tmp_path/'output',frame_index=300)
    assert called['start']==300 and called['end']==301
    assert report['frame']==300 and report['accepted']
    seed=__import__('json').loads((tmp_path/'output/seed.json').read_text())
    assert seed['frame']==300


def test_later_frame_index_validation(tmp_path):
    from real2sim2real.reconstruction.initial_registration import probe
    for frame in (-1, True, 1.5):
        with pytest.raises(ValueError,match='frame_index'):
            probe(None,tmp_path,tmp_path,{},tmp_path/'output',frame_index=frame)
