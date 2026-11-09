import numpy as np
import pytest
from real2sim2real.reconstruction_repair.observed_pose_graph import fit,FitConfig


def scene():
    import trimesh
    v=trimesh.creation.box(extents=[.5,.1,.6]).vertices
    n=9;a=np.tile(np.eye(4),(n,1,1));a[:,2,3]=2
    seed=a.copy();seed[3:,0,3]=.04
    k=np.tile(np.array([[180.,0,100],[0,180,100],[0,0,1]]),(n,1,1))
    angle=np.linspace(0,2*np.pi,16,endpoint=False);axes=np.c_[np.cos(angle),np.sin(angle)]
    camera=v+[0,0,2];uv=camera@k[0].T;uv=uv[:,:2]/uv[:,2:]
    target=np.tile(np.max(uv@axes.T,axis=0),(n,1));weights=np.ones_like(target)
    dp=np.tile(v[None],(n,1,1));dz=dp[:,:,2]+2
    plane=np.tile(np.array([0.,-1.,0.,.1]),(n,1));lock=np.zeros(n,bool);lock[0]=True
    return [v,a,seed,k,np.arange(n)/30,axes,target,weights,dp,dz,np.ones_like(dz),plane,lock]


def test_joint_fit_reduces_source_cost_and_preserves_anchor():
    args=scene();result,receipt=fit(*args,config=FitConfig(max_seconds=5.,max_calls=500,proxy_vertices=32))
    assert receipt['best_proxy_cost']<receipt['initial_proxy_cost']*.1
    np.testing.assert_array_equal(result[0],args[1][0])
    assert np.abs(result[:,0,3]).max()<.005
    assert not receipt['full_reference_approved'] and receipt['independent_native_triangle_depth_support_motion_checks_required']


def test_missing_observation_and_nonuniform_clock_fail_closed():
    args=scene();args[7][4]=0
    with pytest.raises(ValueError):fit(*args)
    args=scene();args[4][3]+=.001
    with pytest.raises(ValueError):fit(*args)


def test_out_of_domain_seed_is_not_clipped_into_a_passing_pose():
    args=scene();args[2][4,0,3]=1
    with pytest.raises(ValueError,match='outside'):fit(*args)
