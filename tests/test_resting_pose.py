import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from real2sim2real.reconstruction_repair.resting_pose import propose

def inputs():
    v=np.array([[-.02,-.02,0],[.02,-.02,0],[.02,.02,0],[-.02,.02,0],[0,0,.1]])
    r=Rotation.from_euler('y',10,degrees=True)
    return v,np.r_[[0,0,0],r.as_quat()],r.apply([0,0,1]),np.array([0,0,1,0]),np.array([0,0,.05])

def test_exact_clearance_and_reviewed_upright_axis():
    v,p,axis,plane,pivot=inputs();old=v.copy()
    q,receipt=propose(v,p,axis,plane,pivot)
    points=Rotation.from_quat(q[3:]).apply(v)+q[:3]
    assert np.isclose(points[:,2].min(),.001)
    assert np.allclose(Rotation.from_quat(q[3:]).apply([0,0,1]),[0,0,1])
    assert np.array_equal(v,old) and receipt['proposal_only']

def test_tangent_translation_keeps_normal_clearance():
    v,p,a,plane,pivot=inputs();q,_=propose(v,p,a,plane,pivot,tangent_offset_m=(.02,-.01))
    assert np.isclose((Rotation.from_quat(q[3:]).apply(v)+q[:3])[:,2].min(),.001)

@pytest.mark.parametrize('kwargs',[dict(max_turn_deg=5),dict(max_translation_m=.0001),dict(fraction=2),dict(clearance_m=-1)])
def test_declared_bounds_are_enforced(kwargs):
    with pytest.raises(ValueError):propose(*inputs(),**kwargs)

def test_nonunit_source_axis_is_rejected():
    v,p,a,plane,pivot=inputs()
    with pytest.raises(ValueError):propose(v,p,a*2,plane,pivot)
