import numpy as np
import pytest
from real2sim2real.retargeting.surface_support import validate_components
from real2sim2real.retargeting.static_support import static_geometry


def box(x,y,z):
    from scipy.spatial import ConvexHull
    points=np.array([[a,b,c] for a in [0,x] for b in [0,y] for c in [0,z]])
    return points,ConvexHull(points).simplices


def test_old_wedges_cannot_pass_support_quality_gate():
    for dims in [(3.66,2.84,.84),(2.81,1.93,.88)]:
        with pytest.raises(ValueError,match='oversized'):validate_components([box(*dims)])


def test_separate_table_parts_remain_admissible():
    report=validate_components([box(.8,.7,.04),box(.06,.07,.7),box(.8,.7,.1)])
    assert report['passed'] and len(report['components'])==3


def test_unobserved_flat_surface_does_not_get_padded():
    y,x=np.indices((48,48));depth=np.full((48,48),2.);k=np.array([[100.,0,24],[0,100,24],[0,0,1]])
    empty=np.zeros(depth.shape,bool)
    with pytest.raises(ValueError):static_geometry(depth,k,empty,empty,[[0,0,48,48]],[0,0,1,-1])


def test_legacy_mode_must_be_explicit():
    d=np.ones((20,20));mask=np.zeros_like(d,bool)
    with pytest.raises(ValueError,match='mode'):static_geometry(d,np.eye(3),mask,mask,[],[0,0,1,0],mode='silent_hull')


def test_observed_side_produces_surface_without_background_hull():
    d=np.ones((64,64));d[10:53,10:53]=2.;d[51:55,10:53]=1.97
    k=np.array([[120.,0,32],[0,120,32],[0,0,1.]])
    human=np.zeros(d.shape,bool);obj=human.copy();obj[27:35,27:35]=True
    p,f,q=static_geometry(d,k,human,obj,[[0,0,64,64]],[0,0,1,-1])
    assert q['vertical_structure_recovered'] is False
    assert np.max(np.ptp(p,axis=0))<1
    assert q['plane']['measured_thickness_m']==pytest.approx(.03,abs=.002)
    assert len(f)>0
