import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from real2sim2real.retargeting.frames import reframe_object


def test_reframing_preserves_every_world_surface_point_without_scaling():
    poses=np.tile(np.eye(4),(3,1,1));poses[:,:3,3]=np.arange(9).reshape(3,3)*.1
    vertices=np.random.default_rng(3).normal(size=(20,3))*.1
    a=np.eye(4);a[:3,:3]=Rotation.from_euler('x',.3).as_matrix();a[:3,3]=[1,2,3]
    b=np.eye(4);b[:3,:3]=Rotation.from_euler('y',.2).as_matrix();b[:3,3]=[.1,-.3,.2]
    changed,rebaked=reframe_object(poses,vertices,a,b)
    for old,new in zip(poses,changed):
        original_world=(vertices@old[:3,:3].T+old[:3,3])@a[:3,:3].T+a[:3,3]
        np.testing.assert_allclose(rebaked@new[:3,:3].T+new[:3,3],original_world,atol=1e-12)


def test_frame_change_rejects_scale():
    a=np.eye(4);a[0,0]=2
    with pytest.raises(ValueError,match='rigid'):
        reframe_object(np.eye(4)[None],np.zeros((4,3)),a,np.eye(4))
