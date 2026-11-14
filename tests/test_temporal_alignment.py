import numpy as np
import cv2
from real2sim2real.egocentric.temporal_alignment import project,static_registration,common_cohort,track_points


def test_static_pose_holdout_separates_camera_motion():
    p=np.random.default_rng(2).uniform([-.3,-.2,.7],[.3,.2,2.],(80,3));k=np.array([[500.,0,320],[0,500,240],[0,0,1]])
    t=np.eye(4);t[:3,3]=[.02,-.01,.005];q=project(p,t,k)
    result,facts=static_registration(p,q,k,np.ones(80,bool),np.arange(80)%2==0)
    assert facts['held_p90_px']<.01
    assert np.allclose(result,t,atol=1e-5)


def test_missing_offsets_cannot_change_identity_denominator():
    assert common_cohort([[True,True,False],[True,False,True]]).tolist()==[True,False,False]
    t,facts=static_registration(np.ones((10,3)),np.ones((10,2)),np.eye(3),np.ones(10,bool),np.arange(10)%2==0)
    assert t is None and not facts['available']


def test_tracks_follow_known_image_shift():
    rng=np.random.default_rng(3);a=rng.integers(0,256,(120,160),dtype=np.uint8);b=cv2.warpAffine(a,np.array([[1.,0,2],[0,1,1]]),(160,120))
    p=np.array([[50,50],[90,70]],np.float32);q,valid,_=track_points(a,b,p)
    assert valid.all()
    assert np.allclose(q-p,[2,1],atol=.2)


def test_patch_motion_uses_same_corners_over_all_offsets():
    from real2sim2real.egocentric.temporal_alignment import track_patch_series
    a=np.random.default_rng(3).integers(0,256,(120,160),dtype=np.uint8)
    images=[cv2.warpAffine(a,np.array([[1.,0,d],[0,1,0]]),(160,120)) for d in [-2,0,2]]
    q,valid,report=track_patch_series(a,images,[70,60])
    assert valid.all() and report['common_corners']>=3
    assert np.allclose(q,[[68,60],[70,60],[72,60]],atol=.3)
