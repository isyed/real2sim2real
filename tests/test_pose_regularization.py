import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from real2sim2real.reconstruction_repair.pose_regularization import propose


def scene():
    n=45;a=np.tile(np.eye(4),(n,1,1));a[:,2,3]=2
    b=a.copy();b[n//2:,:3,:3]=Rotation.from_euler('z',20,degrees=True).as_matrix()
    b[n//2:,0,3]=.08;lock=np.zeros(n,bool);lock[[0,-1]]=True
    return a,b,np.arange(n)/30,lock


def test_step_corrections_become_rigid_prior_with_exact_anchors():
    a,b,t,lock=scene();p,receipt=propose(a,b,t,local_center=np.array([.1,0,0]),locked=lock)
    np.testing.assert_array_equal(p[lock],a[lock])
    angle=np.rad2deg((Rotation.from_matrix(p[1:,:3,:3])*Rotation.from_matrix(p[:-1,:3,:3]).inv()).magnitude())
    assert angle.max()<10
    np.testing.assert_allclose(p[:,:3,:3]@p[:,:3,:3].swapaxes(1,2),np.tile(np.eye(3),(len(p),1,1)),atol=1e-10)
    assert not receipt['source_evidence_validated'] and not receipt['full_reference_approved']


def test_large_change_fails_instead_of_clipping():
    a,b,t,lock=scene()
    with pytest.raises(ValueError,match='uncertainty'):propose(a,b,t,local_center=np.zeros(3),locked=lock,max_center_shift_m=.001)


def test_native_clock_and_explicit_anchor_required():
    a,b,t,lock=scene();t[10]+=.001
    with pytest.raises(ValueError,match='Uniform'):propose(a,b,t,local_center=np.zeros(3),locked=lock)
    with pytest.raises(ValueError):propose(a,b,np.arange(len(a))/30,local_center=np.zeros(3),locked=~np.ones(len(a),bool))


def test_identical_observations_preserve_a_moving_reference():
    a,b,t,lock=scene();a[:,0,3]=np.sin(t)*.2;a[:,:3,:3]=Rotation.from_euler('y',(t*.1)[:,None]).as_matrix()
    p,receipt=propose(a,a,t,local_center=np.ones(3)*.1,locked=lock)
    np.testing.assert_allclose(p,a,atol=1e-12)
