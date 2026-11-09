import numpy as np
import pytest
from real2sim2real.retargeting.segmented_support import component_hull,planar_component_hull


def scene():
    y,x=np.indices((32,32));depth=2+.002*np.sin(x)+.003*np.cos(y)
    depth[:,20:]=6+.002*np.sin(x[:,20:])
    return depth,np.array([[100.,0,16],[0,100,16],[0,0,1]]),np.ones((32,32),bool)


def test_connected_foreground_excludes_background_in_same_mask():
    d,k,m=scene();p,f,q=component_hull(d,k,m,seeds_xy=[(8,8),(10,20)],max_axis_span_m=1)
    assert np.max(p[:,2])<2.01
    assert len(p)==32*20
    assert q['component_fraction']==pytest.approx(20/32)
    assert np.max(f)<len(p)


def test_seeds_may_not_join_two_depth_layers():
    d,k,m=scene()
    with pytest.raises(ValueError,match='one connected'):component_hull(d,k,m,seeds_xy=[(8,8),(28,8)])


def test_unconnected_tiny_seed_is_rejected():
    d,k,m=scene();d[8,8]=4.
    with pytest.raises(ValueError,match='too small'):component_hull(d,k,m,seeds_xy=[(8,8)])


def test_flat_surface_gets_no_invented_thickness():
    d,k,m=scene();d[:]=2.
    with pytest.raises(ValueError,match='invent thickness'):component_hull(d,k,m,seeds_xy=[(8,8)])


def test_metric_bound_rejects_large_connected_region():
    d,k,m=scene()
    with pytest.raises(ValueError,match='metric span'):component_hull(d,k,m,seeds_xy=[(8,8)],max_axis_span_m=.1)


def test_masked_seed_fails_explicitly():
    d,k,m=scene();exclude=np.zeros_like(m);exclude[8,8]=True
    with pytest.raises(ValueError,match='outside'):component_hull(d,k,m,seeds_xy=[(8,8)],exclusion_mask=exclude)


def test_explicit_camera_depth_band_stops_smooth_background_bridge():
    y,x=np.indices((32,32));depth=2+.01*x+.003*np.cos(y)
    k=np.array([[100.,0,16],[0,100,16],[0,0,1]])
    p,_,q=component_hull(depth,k,np.ones((32,32),bool),seeds_xy=[(8,8)],depth_range_m=(1.9,2.15))
    assert np.max(p[:,2])<=2.15
    assert q['selected_points']<32*32


def planar_points(side=True):
    x,y=np.meshgrid(np.linspace(-.3,.3,24),np.linspace(-.25,.25,24))
    top=np.c_[x.ravel(),y.ravel(),(.8+.001*np.sin(x*20)*np.cos(y*20)).ravel()]
    edge=np.c_[np.linspace(-.3,.3,50),np.full(50,-.25),np.full(50,.77)]
    return np.r_[top,edge] if side else top


def test_plane_removes_spikes_and_measures_visible_side():
    vertices,faces,proof=planar_component_hull(planar_points(),expected_up=[0,0,1])
    assert proof['measured_thickness_m']==pytest.approx(.03,abs=.002)
    assert vertices[:,2].max()==pytest.approx(.8,abs=.002)
    assert proof['observed_side_points']>=20
    assert len(faces)>0


def test_plane_without_observed_side_gets_no_added_thickness():
    with pytest.raises(ValueError,match='invent thickness'):
        planar_component_hull(planar_points(False),expected_up=[0,0,1])


def test_vertical_part_is_not_forced_into_tabletop_plane():
    p=planar_points();p=p[:,[2,1,0]]
    with pytest.raises(ValueError):planar_component_hull(p,expected_up=[0,0,1])


def test_nonfinite_bounds_cannot_disable_geometry_gates():
    d,k,m=scene()
    with pytest.raises(ValueError):component_hull(d,k,m,seeds_xy=[(8,8)],max_axis_span_m=float('nan'))
