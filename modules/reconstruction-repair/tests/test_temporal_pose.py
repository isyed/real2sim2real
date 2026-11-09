from dataclasses import replace
import numpy as np
import pytest

from conftest import box_geometry
from real2sim2real.reconstruction_repair.geometry import mesh_silhouette
from real2sim2real.reconstruction_repair.temporal_pose import TemporalPoseConfig,refine_window,validate


def dense_box():
    g=box_geometry(21);v=list(g['vertices']);faces=g['faces']
    for _ in range(2):
        cache={};new=[]
        def midpoint(a,b):
            key=tuple(sorted([int(a),int(b)]))
            if key not in cache:cache[key]=len(v);v.append((np.array(v[a])+v[b])*.5)
            return cache[key]
        for a,b,c in faces:
            ab,bc,ca=midpoint(a,b),midpoint(b,c),midpoint(c,a)
            new.extend([[a,ab,ca],[ab,b,bc],[ca,bc,c],[ab,bc,ca]])
        faces=np.array(new)
    g.update(vertices=np.array(v),faces=faces)
    return g


def fixture():
    g=dense_box();n=21
    masks=np.stack([mesh_silhouette(g,i,96,128) for i in range(n)])
    poses=np.tile(np.eye(4),(n,1,1));poses[:,:3,3]=g['object_translation']
    poses[1:-1,2,3]+=.035*np.sin(np.arange(1,n-1)*2.1)
    return dict(vertices=g['vertices'],faces=g['faces'],poses_camera=poses,intrinsics=g['intrinsics'],
        frame_ids=np.arange(n),timestamps_s=np.arange(n)/30.,object_mask=masks,occluder_mask=np.zeros_like(masks),
        depth_m=np.full(masks.shape,1.97),fit=np.arange(n)%4==0,locked=np.isin(np.arange(n),[0,n-1]),
        support_plane_camera=np.tile([0.,-1.,0.,.052],(n,1)),support_center_camera=np.tile([0.,.052,2.],(n,1)),
        support_radius_m=np.full(n,.5))


CFG=TemporalPoseConfig(knot_stride=5,max_function_calls=120,max_seconds=10,mesh_samples=512,min_depth_points=8)


def test_joint_window_reduces_jitter_and_preserves_anchors():
    data=fixture();original=data['poses_camera'].copy();candidate,r=refine_window(data,CFG)
    assert r['decision']['accepted'],r['decision']
    assert np.array_equal(candidate[data['locked']],original[data['locked']])
    assert np.array_equal(data['poses_camera'],original)
    assert r['decision']['motion_after']['step_p95_m']<r['decision']['motion_before']['step_p95_m']*.8
    assert not r['approved_for_use'] and not r['full_trajectory_accepted'] and not r['interaction_validated']


def test_heldout_images_can_reject_but_cannot_fit():
    data=fixture();a,_=refine_window(data,CFG)
    data['object_mask'][~data['fit']]=np.roll(data['object_mask'][~data['fit']],40,axis=2)
    b,report=refine_window(data,CFG)
    assert np.allclose(a,b,atol=1e-9)
    assert not report['decision']['accepted']
    assert 'image_gate' in report['decision']['reasons']


def test_support_violation_is_not_accepted():
    data=fixture();data['support_plane_camera'][:,3]=-.5
    _,r=refine_window(data,replace(CFG,max_function_calls=20))
    assert not r['decision']['accepted'] and 'support_gate' in r['decision']['reasons']


def test_budget_stops_optimizer_without_claiming_full_reference():
    _,r=refine_window(fixture(),replace(CFG,max_function_calls=1))
    assert r['function_calls']==1 and r['budget_hit']
    assert not r['full_trajectory_accepted']


@pytest.mark.parametrize('problem',['no_anchor','no_holdout','nonfinite','improper_rotation','frame_gap'])
def test_invalid_evidence_stops(problem):
    d=fixture()
    if problem=='no_anchor':d['locked'][:]=False
    if problem=='no_holdout':d['fit'][:]=True
    if problem=='nonfinite':d['depth_m'][0,0,0]=np.nan
    if problem=='improper_rotation':d['poses_camera'][0,0,0]=-1
    if problem=='frame_gap':d['frame_ids'][5:]+=1
    with pytest.raises(ValueError):validate(d)


def test_translation_only_refinement_preserves_every_input_rotation():
    data=fixture()
    from scipy.spatial.transform import Rotation
    data['poses_camera'][:,:3,:3]=Rotation.from_euler('z',(np.sin(np.arange(21)*.7)*.3)[:,None],degrees=True).as_matrix()
    original=data['poses_camera'].copy()
    _,report=refine_window(data,replace(CFG,preserve_rotations=True))
    candidate,_=refine_window(data,replace(CFG,preserve_rotations=True,max_function_calls=1))
    assert report['rotations_preserved_exactly']
    assert np.array_equal(candidate[:,:3,:3],original[:,:3,:3])
    assert np.array_equal(data['poses_camera'],original)


def test_rotation_preservation_option_requires_boolean():
    with pytest.raises(ValueError,match='boolean'):
        replace(CFG,preserve_rotations=1).validate()


def test_smaller_translation_proposals_keep_every_noneditable_frame():
    from real2sim2real.reconstruction_repair.localized_pose import backtrack_translation
    data=fixture();proposal=data['poses_camera'].copy();proposal[:,2,3]-=.02
    editable=np.zeros(21,bool);editable[8:13]=True
    candidate,report=backtrack_translation(data,proposal,CFG,fractions=(.5,),editable=editable)
    assert np.array_equal(candidate[~editable],data['poses_camera'][~editable])
    assert np.array_equal(candidate[:,:3,:3],data['poses_camera'][:,:3,:3])
    assert not report['approved_for_use']


def test_backtracking_rejects_rotational_proposal_and_invalid_fractions():
    from real2sim2real.reconstruction_repair.localized_pose import backtrack_translation
    data=fixture();proposal=data['poses_camera'].copy();proposal[4,:3,:3]=np.diag([-1.,-1.,1.])
    with pytest.raises(ValueError,match='rotations'):backtrack_translation(data,proposal,CFG)
    with pytest.raises(ValueError,match='fractions'):backtrack_translation(data,data['poses_camera'],CFG,fractions=(.25,.5))


def test_interaction_veto_cannot_be_overridden_by_geometric_backtracking():
    from real2sim2real.reconstruction_repair.localized_pose import backtrack_translation
    data=fixture();proposal=data['poses_camera'].copy();proposal[:,2,3]=2.
    candidate,report=backtrack_translation(data,proposal,CFG,fractions=(.5,),extra_gate=lambda p:dict(passed=False))
    assert not report['accepted'] and np.array_equal(candidate,data['poses_camera'])
    assert 'extra_interaction_gate' in report['trials'][0]['decision']['reasons']


def test_existing_positional_knot_stride_argument_keeps_its_meaning():
    assert TemporalPoseConfig(7).knot_stride==7
    assert TemporalPoseConfig(7).preserve_rotations is False
