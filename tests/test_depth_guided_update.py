import numpy as np
import pytest
from real2sim2real.reconstruction.depth_guided_update import correct
from real2sim2real.reconstruction_repair.depth_translation import bounded_proposals


def pose(z):
    p=np.eye(4);p[2,3]=z;return p


V=np.array([[-.1,-.1,0],[.1,.1,0]])


def measure(p, *, iou=.8, points=500, penetration=0):
    delta=2.-p[2,3]
    return dict(original=dict(metrics=dict(iou=iou,precision=.9,recall=.9),
        geometry=dict(depth_points=points,depth_shift_m=delta,depth_error_m=abs(delta),
                      support_penetration_m=penetration,behind_camera=False)))


def run(p,previous=None,measure=measure,**kw):
    return correct(p,pose(2.) if previous is None else previous,vertices=V,
                   source_frame=1,previous_frame=0,fps=30,measure=measure,**kw)


def test_depth_drift_can_be_corrected_without_relaxing_motion():
    p=pose(2.15);original=p.copy();out=run(p)
    assert not out['normal_attempt']['accepted'] and out['accepted']
    assert out['depth_guided_translation']['applied']
    assert abs(np.asarray(out['pose_camera'])[2,3]-2.)<=.08
    assert np.array_equal(p,original)
    assert all(np.array_equal(np.asarray(c['pose_camera'])[:3,:3],p[:3,:3]) for c in out['candidates'])


def test_already_accepted_normal_update_is_unchanged():
    out=run(pose(2.02))
    assert out['accepted'] and out['pose_camera']==pose(2.02).tolist()
    assert not out['depth_guided_translation']['applied'] and not out['depth_guided_translation']['proposals']


def test_large_drift_is_capped_and_still_rejected_against_actual_history():
    out=run(pose(2.5))
    assert not out['accepted'] and out['pose_camera'] is None
    assert all(c['translation_change_m']<=.15000001 for c in out['depth_guided_translation']['proposals'])
    assert all(c['correction_capped'] for c in out['depth_guided_translation']['proposals'])
    assert out['depth_guided_translation']['proposals'][0]['signed_source_surface_residual_m']==-.5


@pytest.mark.parametrize('kwargs',[dict(iou=.3),dict(penetration=.05)])
def test_fitting_depth_cannot_override_independent_image_or_support_failure(kwargs):
    assert not run(pose(2.15),measure=lambda p:measure(p,**kwargs))['accepted']


def test_insufficient_visible_depth_does_not_generate_a_correction():
    out=run(pose(2.15),measure=lambda p:measure(p,points=20))
    assert not out['accepted'] and not out['depth_guided_translation']['proposals']


def test_rotation_is_not_repaired_or_overridden():
    p=pose(2.15);p[:3,:3]=[[0,-1,0],[1,0,0],[0,0,1]]
    assert not run(p)['accepted']


def test_bounded_ray_proposals_preserve_center_projection_and_dont_resize_mesh():
    p=pose(2.15);p[0,3]=.4;original=V.copy();center=p[:3,:3]@V.mean(0)+p[:3,3]
    for row in bounded_proposals(V,p,-.5,max_move_m=.1):
        q=np.asarray(row['pose_camera']);after=q[:3,:3]@V.mean(0)+q[:3,3]
        assert np.allclose(center[:2]/center[2],after[:2]/after[2])
        assert row['translation_change_m']<=.10000001 and not row['candidate_approved']
    assert np.array_equal(V,original)
