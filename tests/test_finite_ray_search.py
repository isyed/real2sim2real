import numpy as np
import pytest
from real2sim2real.reconstruction.depth_guided_update import finite_ray_search, correct
from real2sim2real.reconstruction_repair.depth_translation import finite_ray_proposals

V=np.array([[-.1,-.1,0],[.1,.1,0]])

def pose(z):
    p=np.eye(4);p[2,3]=z;return p

def evidence(p, *, iou=.8, penetration=0):
    # Signed residual can be small despite a broad absolute-error distribution.
    return dict(original=dict(metrics=dict(iou=iou,precision=.9,recall=.9),
        geometry=dict(depth_points=500,depth_shift_m=-.04,
            depth_error_m=abs(p[2,3]-2.)+.06,support_penetration_m=penetration,behind_camera=False)))

def search(p=None,previous=None,measure=evidence):
    return finite_ray_search(pose(2.12) if p is None else p,pose(2.05) if previous is None else previous,
        vertices=V,source_frame=1,previous_frame=0,fps=30,
        signed_distances_m=[0.,-.05,-.10,-.15],measure=measure)

def test_absolute_depth_search_can_pass_when_signed_residual_heuristic_stops_short():
    raw,previous=pose(2.12),pose(2.05)
    assert not correct(raw,previous,vertices=V,source_frame=1,previous_frame=0,fps=30,measure=evidence)['accepted']
    out=search();assert out['accepted']
    assert out['candidates'][out['selected_index']]['evidence']['original']['geometry']['depth_error_m']<.1
    assert np.array_equal(raw,pose(2.12))

@pytest.mark.parametrize('kwargs',[dict(iou=.3),dict(penetration=.05)])
def test_finite_search_does_not_override_image_or_support_veto(kwargs):
    assert not search(measure=lambda p:evidence(p,**kwargs))['accepted']

def test_motion_is_measured_against_accepted_history_before_expensive_measurement():
    calls=[]
    result=search(p=pose(2.5),previous=pose(2.),measure=lambda p:calls.append(p))
    assert not result['accepted'] and not calls
    assert all(c['motion_rejected_for'] for c in result['candidates'])

def test_ray_preserves_rotation_and_projection_without_changing_inputs():
    p=pose(2.);p[0,3]=.4;original=p.copy();v=V.copy()
    for row in finite_ray_proposals(V,p,signed_distances_m=[-.15,0,.15]):
        q=np.asarray(row['pose_camera']);a=p[:3,:3]@V.mean(0)+p[:3,3];b=q[:3,:3]@V.mean(0)+q[:3,3]
        assert np.array_equal(q[:3,:3],p[:3,:3]) and np.allclose(a[:2]/a[2],b[:2]/b[2])
        assert np.linalg.norm(q[:3,3]-p[:3,3])<=.15000001 and not row['candidate_approved']
    assert np.array_equal(p,original) and np.array_equal(v,V)

@pytest.mark.parametrize('distances',[[],[0,0],[float('nan')],[-.151],np.linspace(-.15,.15,65)])
def test_invalid_or_unbounded_grid_is_rejected(distances):
    with pytest.raises(ValueError):finite_ray_proposals(V,pose(2.),signed_distances_m=distances)

def test_search_never_generates_centroid_behind_camera():
    rows=finite_ray_proposals(V,pose(.02),signed_distances_m=[-.15,0,.15])
    assert len(rows)==2 and all(np.asarray(r['pose_camera'])[2,3]>.01 for r in rows)
