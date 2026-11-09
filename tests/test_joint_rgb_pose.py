import copy
import itertools
import numpy as np
import pytest
from real2sim2real.reconstruction_repair.joint_rgb_pose import fit,Config


def problem():
    v=np.array(list(itertools.product([-.12,.12],[-.08,.08],[-.05,.05])))
    t=np.array([0.,.1,.2]);seed=np.tile(np.eye(4),(3,1,1));seed[:,2,3]=3.
    k=np.array([[500.,0,320],[0,500.,240],[0,0,1]])
    a=np.linspace(0,2*np.pi,32,endpoint=False);axes=np.c_[np.cos(a),np.sin(a)];obs=[]
    for shift in [0.,.025,.05]:
        xyz=v+[shift,0,3];uv=xyz@k.T;uv=uv[:,:2]/uv[:,2:]
        obs.append(dict(K=k,directions=axes,outline_px=np.max(uv@axes.T,axis=0),
            outline_observed=np.ones(32,bool),support_plane_camera=np.array([0,-1.,0,1]),
            source_provenance={'rgb':'synthetic'},ground_truth_used=False))
    return [v,t,seed,obs,np.linspace(0,.2,5)]


def test_image_fit_improves_and_keeps_anchor_inputs_and_bounds():
    args=problem();before=copy.deepcopy(args)
    poses,r=fit(*args,locked=[True,False,False],config=Config(max_calls=700,max_seconds=10))
    assert r['best_proxy_cost']<r['initial_proxy_cost'] and abs(poses[-1,0,3]-.05)<.015
    assert np.array_equal(poses[0],args[2][0]) and np.array_equal(args[2],before[2])
    assert r['max_knot_center_shift_m']<=.15 and r['max_knot_rotation_deg']<=25
    assert not r['depth_used_for_fitting'] and not r['heldout_observations_used'] and not r['reference_approved']


def test_depth_cannot_accidentally_enter_rgb_only_fit():
    args=problem();args[3][0]['depth_m']=np.ones((2,2))
    with pytest.raises(ValueError,match='no depth'):fit(*args,locked=[True,False,False])


def test_missing_outline_is_not_treated_as_visible_boundary():
    args=problem();args[3][1]['outline_observed'][:25]=False
    with pytest.raises(ValueError,match='visible outline'):fit(*args,locked=[True,False,False])


def test_missing_anchor_and_extrapolation_rejected():
    with pytest.raises(ValueError):fit(*problem(),locked=[False,False,False])
    args=problem();args[-1]=np.array([0.,.3])
    with pytest.raises(ValueError):fit(*args,locked=[True,False,False])


def test_budget_exhaustion_is_explicit_and_not_acceptance():
    _,r=fit(*problem(),locked=[True,False,False],config=Config(max_calls=1))
    assert r['budget_reached'] and not r['reference_approved']


def test_source_provenance_required():
    args=problem();args[3][0]['source_provenance']={}
    with pytest.raises(ValueError):fit(*args,locked=[True,False,False])


def test_restarts_share_budget_and_never_compound_seed_bounds():
    args=problem();poses,r=fit(*args,locked=[True,False,False],config=Config(restarts=7,max_calls=200))
    assert r['calls']<=200 and r['budget_reached']
    assert r['restart_bounds_relative_to_original_seeds']
    assert r['max_knot_center_shift_m']<=.15 and r['max_knot_rotation_deg']<=25
    assert np.array_equal(poses[0],args[2][0])


def test_withheld_query_sampling_cannot_change_fit_solution():
    args=problem();p1,_=fit(*args,locked=[True,False,False])
    args[-1]=args[1].copy();p2,_=fit(*args,locked=[True,False,False])
    np.testing.assert_allclose(p1[::2],p2,atol=1e-10)


def test_untrusted_seeds_may_move_only_with_explicit_unanchored_request():
    _,r=fit(*problem(),locked=[False,False,False],allow_unanchored=True)
    assert r['unanchored_fit'] and not r['external_seams_validated']
    assert r['locked_knots']==[] and not r['reference_approved']


def test_recovers_rotation_as_well_as_translation_from_image_geometry():
    from scipy.spatial.transform import Rotation
    args=problem()
    for i,obs in enumerate(args[3]):
        xyz=args[0]@Rotation.from_euler('z',i*5,degrees=True).as_matrix().T+[i*.025,0,3]
        uv=xyz@obs['K'].T;uv=uv[:,:2]/uv[:,2:]
        obs['outline_px']=np.max(uv@obs['directions'].T,axis=0)
    poses,r=fit(*args,locked=[True,False,False],config=Config(max_calls=1200))
    assert abs(Rotation.from_matrix(poses[-1,:3,:3]).as_euler('xyz',degrees=True)[2]-10)<2
    assert abs(poses[-1,0,3]-.05)<.015 and not r['reference_approved']
