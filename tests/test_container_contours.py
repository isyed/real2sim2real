import numpy as np
from real2sim2real.egocentric.container_contours import constraint_report,projected_boundary
from real2sim2real.egocentric.container_profile import linear_profile


def setup():
    profile=linear_profile([.04,.05,.17]);pose=np.eye(4);pose[:3,3]=[0,0,.6]
    k=np.array([[600.,0,320],[0,600,240],[0,0,1]])
    points=projected_boundary(profile,pose,k,'bottom')[::20]
    return profile,pose,k,dict(name='bottom',kind='bottom',visibility='visible',pixels=points.tolist(),sigma_px=2,complete=True)


def test_complete_end_recovers_projected_diameter():
    p,t,k,a=setup();r,e=constraint_report(p,t,k,[a]);row=r['rows'][0]
    assert np.allclose(row['observed_ellipse']['diameters_px'],[80,80],atol=.01)
    assert np.max(np.abs(row['diameter_error_px']))<.01
    assert not r['metric_diameter_measured']


def test_partial_contour_penalizes_both_under_and_oversize():
    p,t,k,a=setup();a.update(complete=False,pixels=a['pixels'][:5])
    for factor in [.7,1.3]:
        q=p.copy();q[:4]*=factor;r,e=constraint_report(q,t,k,[a])
        assert r['rows'][0]['observed_to_model_median_px']>10
        assert not r['rows'][0]['diameter_available']


def test_hidden_contour_is_unknown_not_passing_or_completed():
    p,t,k,a=setup();a.update(visibility='occluded',pixels=[])
    r,e=constraint_report(p,t,k,[a])
    assert r['available_constraints']==0 and len(e)==0
    assert not r['rows'][0]['available']


def test_partial_visibility_does_not_penalize_unobserved_remainder():
    p,t,k,a=setup();a.update(complete=False,pixels=a['pixels'][:4])
    r,e=constraint_report(p,t,k,[a])
    assert np.max(abs(e))<1e-8
    assert 'model_to_observed_p90_px' not in r['rows'][0]


def test_shared_fitter_accepts_visible_contour_constraints():
    from scipy.spatial.transform import Rotation
    from real2sim2real.egocentric.container_profile import fit_shared, surface
    p,_,k,_=setup();views=[]
    for angle in [35,65]:
        t=np.eye(4);t[:3,:3]=Rotation.from_euler('x',angle,degrees=True).as_matrix();t[:3,3]=[0,.1,.7]
        xyz=surface(p)[::20]@t[:3,:3].T+t[:3,3]
        from real2sim2real.egocentric.container_axis import endpoints
        boundary=projected_boundary(p,t,k,'bottom')[::20]
        annotation=dict(name='visible_bottom',kind='bottom',visibility='visible',complete=True,sigma_px=2,pixels=boundary.tolist())
        views.append(dict(points=xyz,k=k,pixels=endpoints(t,p[4],k),sigma=[2,2],pose=t,image_constraints=[annotation]))
    initial=p.copy();initial[:4]*=.9
    result=fit_shared(views,initial)
    assert result['optimizer_success']
    assert abs(result['profile_m'][0]-p[0])<.003


def test_support_translation_preserves_plane_and_masks_hidden_endpoint():
    from scipy.spatial.transform import Rotation
    from real2sim2real.egocentric.container_profile import fit_shared, surface
    from real2sim2real.egocentric.container_axis import endpoints
    p,_,k,_=setup()
    t=np.eye(4);t[:3,:3]=Rotation.from_euler('x',140,degrees=True).as_matrix();t[:3,3]=[0,.1,.7]
    xyz=surface(p)[::16]@t[:3,:3].T+t[:3,3]
    pixels=endpoints(t,p[4],k);pixels[0]=[-10000.,-10000.]
    supported=dict(points=xyz,k=k,pixels=pixels,sigma=[1,2],endpoint_visible=[False,True],pose=t,pose_mode='support_translation')
    fixed=dict(points=xyz,k=k,pixels=endpoints(t,p[4],k),sigma=[2,2],pose=t,pose_mode='fixed')
    result=fit_shared([supported,fixed],p)
    assert result['optimizer_success']
    fitted=np.array(result['training_poses'][0]);delta=fitted[:3,3]-t[:3,3]
    assert np.allclose(fitted[:3,:3],t[:3,:3])
    assert abs(delta@t[:3,2])<1e-10
    assert np.allclose(result['training_poses'][1],t)
    assert np.allclose(result['profile_m'],p,atol=.003)


def test_fixed_shape_pose_fitter_uses_contour_adapter():
    from scipy.spatial.transform import Rotation
    from real2sim2real.egocentric.container_axis import endpoints, fit_endpoint_pose
    from real2sim2real.egocentric.container_profile import surface,distance
    p,_,k,_=setup();t=np.eye(4);t[:3,:3]=Rotation.from_euler('x',50,degrees=True).as_matrix();t[:3,3]=[0,.1,.7]
    xyz=surface(p)[::16]@t[:3,:3].T+t[:3,3]
    annotation=dict(name='bottom',kind='bottom',visibility='visible',complete=True,sigma_px=2,pixels=projected_boundary(p,t,k,'bottom')[::20].tolist())
    pose,report=fit_endpoint_pose(xyz,[p[0],p[3],p[4]],k,endpoints(t,p[4],k),[2,2],t,
                                surface_distance=lambda x,d:distance(x,p),image_constraints=[annotation],projection_profile=p)
    assert report['visible_contours_used_for_pose']
    assert np.linalg.norm(pose[:3,3]-t[:3,3])<.003
