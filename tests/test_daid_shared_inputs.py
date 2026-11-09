import numpy as np
import pytest
from real2sim2real.reconstruction.daid_inputs import camera_pointmap,recover_original_camera_pose


def test_native_pixel_projection_survives_camera_axis_conversion():
    K=np.array([[800.,0,2.],[0,810.,1.],[0,0,1.]])
    d=np.arange(12,dtype=float).reshape(3,4)/10+2
    p=camera_pointmap(d,K)*[-1,-1,1];xy=p@K.T;xy=xy[...,:2]/xy[...,2:]
    y,x=np.indices(d.shape);np.testing.assert_allclose(xy,np.stack([x,y],-1),atol=1e-5)


def test_unobserved_depth_is_not_filled():
    p=camera_pointmap(np.array([[0.,np.nan],[-1.,2.]]),np.eye(3))
    assert p.dtype==np.float32  # Native PyTorch3D metric normalizer requires float32.
    assert np.isnan(p[0]).all() and np.isnan(p[1,0]).all() and np.isfinite(p[1,1]).all()


@pytest.mark.parametrize('K',[np.zeros((3,3)),np.eye(4),np.array([[1.,1,0],[0,1,0],[0,0,1]])])
def test_bad_calibration_rejected(K):
    with pytest.raises(ValueError):camera_pointmap(np.ones((2,2)),K)


def test_canonical_pose_inverse_matches_original_mesh():
    Q=np.array([[1.,0,0],[0,0,1],[0,-1,0]])
    D=np.diag([-1.,-1.,1.]);Rc=np.eye(3);tc=np.array([.1,-.2,3.])
    center=np.array([.03,-.05,.02]);normalization=4.;scale=1/normalization
    R=Q.T@Rc.T@D;t=(tc+center@Q.T)@D
    result=recover_original_camera_pose(R,t,[scale]*3,Q,normalization,center)
    np.testing.assert_allclose(result[:3,:3],Rc,atol=1e-10)
    np.testing.assert_allclose(result[:3,3],tc,atol=1e-10)
