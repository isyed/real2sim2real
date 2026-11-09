import numpy as np
import pytest

from real2sim2real.reconstruction_repair.interaction_translation import (
    propose, surface_witnesses, InteractionTranslationConfig,
)


def case():
    poses=np.tile(np.eye(4),(7,1,1))
    poses[:,0,3]=[0,.07,-.07,.07,-.07,.07,0]
    times=np.arange(7)/30
    locked=np.isin(np.arange(7),[0,6])
    landmarks=np.tile([[-.5,0,0],[.5,0,0]],(7,1,1))
    witnesses=landmarks+poses[:,None,:3,3]
    distances=np.linalg.norm(witnesses-landmarks,axis=2)
    return poses,times,locked,landmarks,witnesses,distances


def test_joint_projection_removes_jumps_without_moving_locked_poses_or_rotations():
    args=case();before=args[0].copy();candidate,report=propose(*args)
    assert report['feasible_proposal_found']
    assert report['maximum_step_m']<=.08
    assert report['maximum_witness_gap_increase_m']<=.005
    np.testing.assert_array_equal(candidate[args[2]],before[args[2]])
    np.testing.assert_array_equal(candidate[:,:3,:3],before[:,:3,:3])
    np.testing.assert_array_equal(args[0],before)
    assert report['native_observation_validation_required']
    assert not report['full_reference_approved'] and not report['physical_contact_established']


def test_incompatible_witnesses_do_not_become_an_accepted_repair():
    args=list(case());args[4]=args[3].copy();args[5]=np.zeros((7,2))
    candidate,report=propose(*args)
    assert not report['feasible_proposal_found'] and report['original_retained']
    np.testing.assert_array_equal(candidate,args[0])


def test_surface_witness_is_a_point_on_the_transformed_mesh_and_bounds_true_distance():
    import trimesh
    mesh=trimesh.creation.box(extents=[1,.8,.6]);p=np.tile(np.eye(4),(3,1,1))
    p[:,:3,:3]=[[0,-1,0],[1,0,0],[0,0,1]];p[:,:3,3]=[.2,.3,2]
    h=np.tile([[-.3,.3,2],[.2,.9,2]],(3,1,1))
    q,d=surface_witnesses(mesh.vertices,mesh.faces,p,h)
    np.testing.assert_allclose(d,.1,atol=1e-8)
    np.testing.assert_allclose(np.linalg.norm(q-h,axis=2),d)
    delta=np.array([.01,-.01,.02]);moved=trimesh.Trimesh(mesh.vertices@p[0,:3,:3].T+p[0,:3,3]+delta,mesh.faces,process=False)
    _,true_distance,_=trimesh.proximity.closest_point(moved,h[0])
    assert np.all(true_distance<=np.linalg.norm(q[0]+delta-h[0],axis=1)+1e-9)


@pytest.mark.parametrize('problem',['time','gap','rotation','locks'])
def test_invalid_evidence_is_rejected(problem):
    args=list(case())
    if problem=='time':args[1][3]=args[1][2]
    elif problem=='gap':args[5][2,0]+=.03
    elif problem=='rotation':args[0][2,0,0]=-1
    elif problem=='locks':args[2]=np.zeros(7,bool)
    with pytest.raises(ValueError):propose(*args)


def test_time_limit_returns_only_an_explicitly_feasible_proposal():
    args=case();_,report=propose(*args,config=InteractionTranslationConfig(max_seconds=1e-10))
    assert report['deadline_reached'] and report['original_retained']
    assert not report['feasible_proposal_found']
