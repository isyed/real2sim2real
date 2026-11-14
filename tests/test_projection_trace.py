import numpy as np
from real2sim2real.egocentric.camera import project_depth
from real2sim2real.egocentric.projection_trace import project_with_source


def test_trace_uses_nearest_source_and_leaves_holes_unknown():
    p=np.array([[[0,0,2.],[0,0,1.],[np.nan,0,1.],[0,0,-1.]]])
    result=project_with_source(p,np.ones((1,4),bool),np.eye(4),np.eye(3),(3,2))
    assert result['source_index'][0,0]==1
    assert (result['source_index'][1]==-1).all()
    assert np.array_equal(result['depth'],project_depth(p,np.eye(3),(3,2)))


def test_trace_matches_metric_projection_with_rounding_and_rigid_translation():
    rng=np.random.default_rng(4);p=rng.uniform([-.2,-.1,.3],[.2,.1,1],(20,30,3))
    k=np.array([[100,0,32],[0,100,24],[0,0,1.]])
    t=np.eye(4);t[0,3]=.04
    result=project_with_source(p,np.ones((20,30),bool),t,k,(64,48))
    target=p.reshape(-1,3)+t[:3,3]
    assert np.array_equal(result['depth'],project_depth(target,k,(64,48)))
    y,x=np.nonzero(result['source_index']>=0)
    q=target[result['source_index'][y,x]]@k.T
    assert np.allclose(result['target_subpixel'][y,x],q[:,:2]/q[:,2:])
