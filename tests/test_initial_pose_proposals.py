import numpy as np
import pytest
from real2sim2real.reconstruction.initial_registration import proposal_poses, probe


def test_external_proposals_are_bounded_rigid_camera_poses(tmp_path):
    p = tmp_path/'poses.npz'
    np.savez(p, poses_camera=np.eye(4)[None])
    assert proposal_poses(p).shape == (1, 4, 4)
    for poses in [np.repeat(np.eye(4)[None], 33, axis=0), np.full((1,4,4), np.nan), np.zeros((1,4,4))]:
        np.savez(p, poses_camera=poses)
        with pytest.raises(ValueError, match='rigid'):
            proposal_poses(p)


def test_invalid_proposal_fails_before_creating_output_or_loading_model(tmp_path):
    p = tmp_path/'poses.npz'; np.savez(p, poses_camera=np.zeros((1,4,4)))
    output = tmp_path/'out'
    with pytest.raises(ValueError, match='rigid'):
        probe(None, tmp_path/'export', tmp_path/'depth',
              dict(ground_truth_used=False,coordinate_frame='camera',evidence='predicted'),
              output, candidate_poses=p)
    assert not output.exists()


def test_external_hypotheses_are_rendered_and_depth_veto_still_applies(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import real2sim2real.reconstruction.initial_registration as module
    candidates=np.repeat(np.eye(4)[None],2,axis=0);candidates[:,2,3]=[1.,2.]
    path=tmp_path/'poses.npz';np.savez(path,poses_camera=candidates)
    rendered=[]
    class Model:
        mesh_ori=SimpleNamespace(vertices=np.zeros((3,3)))
        def get_tf_to_centered_mesh(self):return np.eye(4)
        def register(self,*args,**kwargs):raise AssertionError('External hypotheses must not rerun registration')
    upstream=SimpleNamespace(FoundationPose=Model,_tensor_to_numpy=np.asarray,
        _foundationpose_frame_iterator=lambda:iter([(0,None,np.ones((2,2)),np.zeros((2,2),bool),np.ones((2,2),bool),np.eye(3))]),
        FOUNDATIONPOSE_SEQUENCE_MODE_REGISTER_FIRST_THEN_TRACK='tracking')
    def render(model,poses,K,shape):rendered.append(poses.copy());return None
    upstream._render_candidate_masks=render
    upstream.measure_visible_silhouettes=lambda *args:None
    upstream._selected_metric_values=lambda measured,i:dict(iou=.99 if i==0 else .8,precision=.9,recall=.9)
    def execute(*args,**kwargs):
        list(upstream._foundationpose_frame_iterator())
        upstream.FoundationPose().register()
    upstream.run_foundationpose_mhr_export=execute
    monkeypatch.setattr(module,'sample_vertices',lambda v:v)
    monkeypatch.setattr(module,'consistency',lambda v,p,*a:dict(depth_points=200,depth_error_m=.3 if p[2,3]==1 else .01,support_penetration_m=0.,behind_camera=False))
    report=probe(upstream,tmp_path/'export',tmp_path/'depth',dict(ground_truth_used=False,coordinate_frame='camera',evidence='own depth'),tmp_path/'out',candidate_poses=path)
    assert report['accepted'] and report['selected_index']==1
    assert report['candidates'][0]['rejected_for']==['depth']
    assert report['proposal_sha256'] and not report['thresholds_relaxed']
    assert np.array_equal(rendered[0],candidates)
    assert upstream.FoundationPose is Model
