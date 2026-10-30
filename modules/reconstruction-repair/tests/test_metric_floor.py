import numpy as np
import pytest
from real2sim2real.reconstruction_repair.metric_floor import fit_floor_plane,PlaneFitError,floor_points


def test_depth_outliers_do_not_move_a_consistent_two_region_floor():
    rng=np.random.default_rng(4);x,z=rng.uniform(-1,1,6000),rng.uniform(1.5,4.5,6000)
    y=(1.5+.02*x-.1*z)/.98+rng.normal(0,.003,6000);y[::10]+=.5
    plane,report=fit_floor_plane(np.c_[x,y,z],np.arange(6000)%2)
    expected=np.array([.02,-.98,-.1,1.5])/np.linalg.norm([.02,-.98,-.1])
    np.testing.assert_allclose(plane,expected,atol=.0005)
    assert report['passed'] and report['inlier_rms_m']<.004


def test_incompatible_floor_regions_remain_blocked():
    rng=np.random.default_rng(0);p=np.c_[rng.uniform(-2,2,4000),np.ones(4000),rng.uniform(1,5,4000)];regions=np.arange(4000)%2;p[regions==1,1]=1.5
    with pytest.raises(PlaneFitError):fit_floor_plane(p,regions)


def test_floor_regions_cannot_silently_resize_native_depth():
    with pytest.raises(ValueError,match='dimensions'):
        floor_points(np.ones((10,10)),np.eye(3),np.zeros((10,10),bool),np.zeros((10,10),bool),
            dict(image_size=[20,20],rectangles_xyxy=[[0,0,5,5],[5,5,9,9]]))
