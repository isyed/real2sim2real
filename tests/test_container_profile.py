import numpy as np
import pytest
from real2sim2real.egocentric.container_profile import distance, surface, linear_profile, fit_shared
from real2sim2real.egocentric.tapered_container import outer_distance
from real2sim2real.egocentric.container_axis import endpoints
from scipy.spatial.transform import Rotation


def test_linear_profile_matches_frozen_frustum_and_keeps_mouth_open():
    dims=[.04,.06,.18];a=linear_profile(dims)
    p=np.random.default_rng(1).uniform([-.1,-.1,-.03],[.1,.1,.24],(100,3))
    assert np.allclose(distance(p,a),outer_distance(p,dims),atol=1e-12)
    assert np.max(distance(surface(a),a))<1e-10
    assert distance(np.array([[0.,0,.18]]),a)[0]>.05


def test_shared_fit_recovers_height_from_two_endpoint_views():
    truth=linear_profile([.04,.055,.16]);k=np.array([[600.,0,640],[0,600,400],[0,0,1]])
    views=[]
    for angle in [65,145]:
        t=np.eye(4);t[:3,:3]=Rotation.from_euler('x',angle,degrees=True).as_matrix();t[:3,3]=[.05,.1,.7]
        p=surface(truth)[::12]@t[:3,:3].T+t[:3,3]
        views.append(dict(points=p,k=k,pixels=endpoints(t,.16,k),sigma=[2,2],pose=t))
    result=fit_shared(views,linear_profile([.04,.055,.18]))
    assert result['optimizer_success']
    assert abs(result['profile_m'][-1]-.16)<.004
    assert not result['control_ready']


def test_invalid_profile_or_missing_views_rejected():
    with pytest.raises(ValueError):surface([.04,np.nan,.05,.06,.2])
    with pytest.raises(ValueError):fit_shared([],linear_profile([.04,.06,.2]))


def test_residual_breakdown_keeps_bad_wall_points_and_separate_end_regions():
    from real2sim2real.egocentric.container_profile import residual_breakdown
    a=linear_profile([.04,.04,.2])
    p=np.array([[.04,0,.1],[.06,0,.1],[0,0,0],[.04,0,.19],[.04,0,.25]])
    r=residual_breakdown(p,a)
    assert sum(v['points'] for v in r.values())==5
    assert r['middle_wall']['points']==2
    assert r['middle_wall']['over_10mm_points']==1
    assert r['outside_height']['points']==1
    assert r['bottom_band']['median_m']==0
