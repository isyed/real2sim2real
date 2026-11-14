import numpy as np
import cv2
import pytest
from real2sim2real.egocentric.camera_orientation import turn_pixel_calibration,convention_streams,pair_geometry


@pytest.mark.parametrize('count',[4,5,8,12,14])
def test_turn_matches_actual_raster_rotation_including_nonradial_distortion(count):
    k=np.array([[500.,0,310],[0,505,235],[0,0,1]])
    d=np.array([.05,-.01,.001,-.002,.005,.02,-.003,.001,.002,-.001,.001,.003,.01,-.008])[:count]
    p=np.random.default_rng(3).uniform([-.3,-.2,.8],[.3,.2,2],(100,3))
    uv=cv2.projectPoints(p,np.zeros(3),np.zeros(3),k,d)[0][:,0]
    newk,newd=turn_pixel_calibration(k,d,[640,480])
    turned=cv2.projectPoints(p*[-1,-1,1],np.zeros(3),np.zeros(3),newk,newd)[0][:,0]
    assert np.allclose(turned,[639,479]-uv,atol=1e-8)
    backk,backd=turn_pixel_calibration(newk,newd,[640,480])
    assert np.allclose(backk,k) and np.allclose(backd,d)


def test_cheirality_rejects_reversed_baseline_even_with_perfect_epipolar_error():
    k=np.array([[500.,0,320],[0,500,240],[0,0,1]])
    a=dict(intrinsics=k.tolist(),distortion=[0.]*5,size=[640,480],rig_from_camera=np.eye(4).tolist())
    b=dict(a);t=np.eye(4);t[0,3]=.075;b['rig_from_camera']=t.tolist()
    p=np.random.default_rng(4).uniform([-.2,-.1,.5],[.2,.1,2],(40,3))
    def image(p):q=p@k.T;return q[:,:2]/q[:,2:]
    uv,v=image(p),image(p-[.075,0,0])
    good=pair_geometry(a,b,uv,v)
    turned=convention_streams(dict(a=a,b=b),axes=True)
    bad=pair_geometry(turned['a'],turned['b'],uv,v)
    assert good['positive_depth_fraction']==1
    assert bad['sampson_median_px']<1e-8 and bad['positive_depth_fraction']==0
