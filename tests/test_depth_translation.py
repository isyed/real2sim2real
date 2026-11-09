import numpy as np
import pytest
from real2sim2real.reconstruction_repair.depth_translation import propose


def test_ray_repair_preserves_rotation_mesh_and_centroid_projection():
    vertices=np.array([[0,0,0],[.1,0,0],[0,.1,0],[0,0,.1]])
    original=vertices.copy();p=np.eye(4);p[:3,3]=[-.2,.3,3.]
    q,proof=propose(vertices,p,-.16)
    a=p[:3,:3]@vertices.mean(0)+p[:3,3];b=q[:3,:3]@vertices.mean(0)+q[:3,3]
    assert np.allclose(a[:2]/a[2],b[:2]/b[2])
    assert np.array_equal(q[:3,:3],p[:3,:3]) and np.array_equal(vertices,original)
    assert proof['translation_change_m']<.15 and not proof['interpolation_used']
    assert not proof['candidate_approved'] and proof['depth_is_prediction']


def test_inside_margin_is_identity_and_excessive_correction_rejected():
    v=np.eye(3);p=np.eye(4);p[2,3]=3.
    q,_=propose(v,p,.02);assert np.array_equal(q,p)
    with pytest.raises(ValueError):propose(v,p,-.8)
    with pytest.raises(ValueError):propose(v,p,float('nan'))
