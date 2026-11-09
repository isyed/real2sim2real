import numpy as np
import pytest
from real2sim2real.reconstruction_repair.support_surface import project_below_support


def inputs():
    # Outward-oriented closed tetrahedron straddling y=0 in a +z camera.
    v=np.array([[0.,-.1,0.],[-.1,.1,-.1],[.1,.1,-.1],[0.,.1,.1]])
    f=np.array([[0,1,2],[0,3,1],[0,2,3],[1,3,2]])
    p=np.eye(4);p[:3,3]=[0.,0.,2.]
    return v,f,p,np.array([0.,-1.,0.,.05])


def test_projection_topology_and_inputs_are_preserved():
    v,f,p,plane=inputs();original=v.copy()
    out,r=project_below_support(v,f,p,plane,resting_evidence='Reviewed stationary support contact',
        max_displacement_m=2.,max_changed_fraction=1.,max_relative_volume_change=.99)
    assert r['numerical_bounds_passed']
    a=v+p[:3,3];b=out+p[:3,3]
    np.testing.assert_allclose(a[:,:2]/a[:,2:],b[:,:2]/b[:,2:],atol=1e-15)
    np.testing.assert_array_equal(v,original)
    np.testing.assert_allclose(out[0],v[0],atol=1e-15)
    assert np.min(b@plane[:3]+plane[3])==pytest.approx(.001)
    assert not r['complete_reference_approved'] and not r['source_depth_validated']


def test_large_proposal_is_reported_rejected_without_clipping():
    v,f,p,plane=inputs()
    _,r=project_below_support(v,f,p,plane,resting_evidence='Reviewed support',max_displacement_m=.001)
    assert not r['numerical_bounds_passed'] and 'displacement_bound' in r['reasons']
    assert 'changed_fraction_bound' in r['reasons']


def test_noop_on_clear_mesh():
    v,f,p,plane=inputs();plane[3]=.3
    out,r=project_below_support(v,f,p,plane,resting_evidence='Reviewed support')
    np.testing.assert_allclose(out,v,atol=1e-15)
    assert r['numerical_bounds_passed'] and r['changed_vertices']==0


def test_float32_pose_keeps_untouched_vertices_exact():
    from scipy.spatial.transform import Rotation
    v,f,p,plane=inputs();p[:3,:3]=Rotation.from_euler('z',13,degrees=True).as_matrix().astype(np.float32)
    camera=v@p[:3,:3].T+p[:3,3];unchanged=camera@plane[:3]+plane[3]>=.001-1e-10
    out,r=project_below_support(v,f,p,plane,resting_evidence='Reviewed contact',max_displacement_m=2.,
        max_changed_fraction=1.,max_relative_volume_change=.99)
    np.testing.assert_array_equal(out[unchanged],v[unchanged])
    assert r['max_normalized_projection_error']<1e-14


def test_normal_projection_is_shortest_displacement_and_changes_outline():
    v,f,p,plane=inputs()
    kwargs=dict(resting_evidence='Reviewed support',max_displacement_m=2.,
        max_changed_fraction=1.,max_relative_volume_change=.99)
    ray,ray_report=project_below_support(v,f,p,plane,**kwargs)
    out,r=project_below_support(v,f,p,plane,projection_mode='support_normal',**kwargs)
    before=v+p[:3,3];after=out+p[:3,3]
    changed=before@plane[:3]+plane[3]<.001
    np.testing.assert_array_equal(out[~changed],v[~changed])
    expected=np.maximum(0,.001-(before@plane[:3]+plane[3]))[:,None]*plane[:3]
    np.testing.assert_allclose(after-before,expected,atol=1e-15)
    assert r['maximum_displacement_m']<ray_report['maximum_displacement_m']
    assert r['numerical_bounds_passed'] and not r['initial_projection_preserved']
    assert r['minimum_ray_factor'] is None and r['max_normalized_projection_error']>0
    assert not r['heldout_image_validated'] and not r['complete_reference_approved']


def test_normal_projection_rejects_excessive_geometry_change():
    v,f,p,plane=inputs()
    _,r=project_below_support(v,f,p,plane,projection_mode='support_normal',
        resting_evidence='Reviewed support',max_displacement_m=.001)
    assert not r['numerical_bounds_passed'] and 'displacement_bound' in r['reasons']


def test_unknown_projection_mode_rejected():
    v,f,p,plane=inputs()
    with pytest.raises(ValueError):
        project_below_support(v,f,p,plane,resting_evidence='Reviewed support',projection_mode='unknown')


@pytest.mark.parametrize('kind',['evidence','plane','camera','topology','limits'])
def test_invalid_inputs_fail(kind):
    v,f,p,plane=inputs();e='Reviewed contact';limit=.1
    if kind=='evidence':e=''
    if kind=='plane':plane*=2
    if kind=='camera':p[2,3]=-.1
    if kind=='topology':f[0,0]=99
    if kind=='limits':limit=float('nan')
    with pytest.raises(ValueError):project_below_support(v,f,p,plane,resting_evidence=e,max_displacement_m=limit)
