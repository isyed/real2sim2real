import numpy as np
import pytest
from real2sim2real.reconstruction_repair.foreground_depth import crop_camera,floor_align


def test_crop_preserves_focal_length_and_rejects_wrong_principal_point():
    K=np.array([[100.,0,80],[0,100,60],[0,0,1]])
    camera,fov=crop_camera(K,[40,20,120,100],(120,160))
    assert np.array_equal(camera[:2,2],[40,40]) and camera[0,0]==100
    assert 43<fov<44
    with pytest.raises(ValueError):crop_camera(K,[39,20,120,100],(120,160))


def fixture():
    K=np.array([[100.,0,80],[0,100,0],[0,0,1]])
    yy,xx=np.mgrid[:120,:160];z=100/np.maximum(yy,1)
    return z,K,np.array([0.,-1.,0.,1.]),[[0,60,60,120],[100,60,160,120]],np.zeros_like(z,bool)


def test_scale_comes_from_floor_not_corrupt_object_depth():
    z,K,p,rect,excluded=fixture();z=z/1.25;z[30:50,65:95]=99
    calibrated,r=floor_align(z,K,p,rect,excluded)
    assert r['accepted'] and abs(r['gain']-1.25)<1e-8
    assert calibrated[40,80]>90 and not r['object_pose_used']


def test_heldout_floor_can_reject_without_changing_fitted_gain():
    z,K,p,rect,excluded=fixture();yy,xx=np.indices(z.shape)
    z[(xx+yy)%2==1]*=1.5
    _,r=floor_align(z,K,p,rect,excluded)
    assert r['gain']==1 and not r['accepted'] and 'heldout_floor_residual' in r['reasons']


def test_missing_floor_does_not_invent_scale():
    z,K,p,rect,excluded=fixture();excluded[:]=True
    result,r=floor_align(z,K,p,rect,excluded)
    assert not r['accepted'] and r['gain'] is None and np.array_equal(result,z)
