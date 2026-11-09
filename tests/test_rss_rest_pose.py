import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from real2sim2real.rss_bridge.bundle import rigid
from real2sim2real.rss_bridge.rest_pose import blend_stationary_tail, fit_pose


def test_rest_blend_preserves_prefix_and_locks_tail():
    p = np.tile(np.eye(4), (10, 1, 1))
    p[:, 0, 3] = np.arange(10)*.001
    rest = np.eye(4)
    rest[:3, :3] = Rotation.from_euler("y", .2).as_matrix()
    rest[:3, 3] = [.02, .03, 3.]
    result = blend_stationary_tail(p, rest, start=3, fully_resting=7)
    np.testing.assert_array_equal(result[:4], p[:4])
    np.testing.assert_array_equal(result[7:], np.tile(rest, (3, 1, 1)))
    rigid(result)
    assert 0 < Rotation.from_matrix(result[5, :3, :3]).magnitude() < .2


@pytest.mark.parametrize("start,end", [(3,3),(-1,4),(1,10),(5,3)])
def test_invalid_transition_rejected(start,end):
    with pytest.raises(ValueError):
        blend_stationary_tail(np.tile(np.eye(4),(10,1,1)),np.eye(4),start=start,fully_resting=end)


def test_fit_budget_and_bounds_are_enforced():
    poses=[]
    mask=np.zeros((40,40),bool);mask[10:30,10:30]=True
    def render(p):
        poses.append(p.copy())
        return mask
    vertices=np.array([[0.,0.,0.],[.01,0.,0.],[0.,.01,0.],[0.,0.,.01]])
    seed=np.eye(4);seed[2,3]=3
    _,receipt=fit_pose(render,seed,vertices,np.stack([mask,mask]),max_evaluations=8)
    assert len(poses)<=8 and receipt['evaluations']==len(poses)
    center=vertices.mean(0)
    for p in poses:
        assert Rotation.from_matrix(p[:3,:3]).magnitude()<=np.deg2rad(15)+1e-7
        delta=p[:3,:3]@center+p[:3,3]-(center+seed[:3,3])
        assert np.all(np.abs(delta)<=np.array([.02,.02,.06])+1e-7)


def test_fit_requires_multiple_visible_observations():
    with pytest.raises(ValueError):
        fit_pose(lambda p:None,np.eye(4),np.ones((4,3)),np.zeros((1,40,40),bool))


def test_locked_center_depth_is_preserved_during_every_render():
    vertices=np.array([[0.,0.,0.],[.1,0.,0.],[0.,.1,0.],[0.,0.,.1]])
    seed=np.eye(4);seed[2,3]=3.
    mask=np.ones((20,20),bool);seen=[];center=vertices.mean(0)
    def render(p):
        seen.append((p[:3,:3]@center+p[:3,3])[2])
        return mask
    _,receipt=fit_pose(render,seed,vertices,np.stack([mask,mask]),
                        preserve_center_depth=True,max_evaluations=15)
    np.testing.assert_allclose(seen,3.+center[2],atol=1e-14,rtol=0)
    assert receipt['camera_center_depth_preserved']
