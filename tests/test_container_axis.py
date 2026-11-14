import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from real2sim2real.egocentric.container_axis import endpoints, fit_endpoint_pose, axis_transition, exclusion_bands
from real2sim2real.egocentric.tapered_container import surface


def test_labeled_endpoints_recover_from_inverted_initial_pose():
    dims=[.04,.055,.18];k=np.array([[600.,0,640],[0,600,400],[0,0,1]])
    pose=np.eye(4);pose[:3,:3]=Rotation.from_euler('x',70,degrees=True).as_matrix();pose[:3,3]=[.05,.1,.7]
    p=surface(*dims)[::5]@pose[:3,:3].T+pose[:3,3]
    initial=pose.copy();initial[:3,3]+=pose[:3,2]*dims[2];initial[:3,:3]=pose[:3,:3]@Rotation.from_euler('x',180,degrees=True).as_matrix()
    fitted,result=fit_endpoint_pose(p,dims,k,endpoints(pose,dims[2],k),[4,4],initial)
    assert fitted[:3,2]@pose[:3,2]>.99
    assert max(result['endpoint_fit_error_px'])<1


def test_axis_comparison_ignores_yaw_but_detects_inversion():
    a=np.eye(4);b=np.eye(4);b[:3,:3]=Rotation.from_euler('z',120,degrees=True).as_matrix()
    assert axis_transition(a,b,1)['axis_angle_deg']==0
    b[:3,:3]=Rotation.from_euler('x',180,degrees=True).as_matrix()
    assert axis_transition(a,b,2)['mean_axis_rate_deg_s']==90
    with pytest.raises(ValueError):axis_transition(a,b,0)


def test_unknown_protrusions_are_not_cleared_and_bands_cover_yaw():
    assert not exclusion_bands(np.empty((0,3)),[.04,.06,.2])['available']
    band=exclusion_bands(np.array([[.065,0,.1],[0,.062,.11]]),[.04,.06,.2])
    assert band['azimuth']=='all'
    assert np.isclose(band['outer_radius_m'],.08)
    assert band['z_min_m']<.1 and band['z_max_m']>.11
    assert not band['geometry_verified']
