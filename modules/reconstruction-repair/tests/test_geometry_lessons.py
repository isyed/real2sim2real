"""Generic regressions motivated by small-object small-object repair failures."""
import numpy as np
from scipy.spatial.transform import Rotation
from real2sim2real.reconstruction_repair.geometry import frame_overlap
from real2sim2real.reconstruction_repair.repairs import limit_pose_motion, pose_recovery_propose
from real2sim2real.reconstruction_repair.scoring import motion_statistics, geometry_scores
from real2sim2real.reconstruction_repair.contracts import RepairProposal
from real2sim2real.reconstruction_repair.validation import proposal_validate
from conftest import box_geometry, observations, only


def test_object_crop_keeps_small_object_pixels_and_full_grid_iou():
    truth=box_geometry(1)
    o=observations(truth)
    # Pad the image while preserving the exact physical projection/observations.
    o['object_mask']=np.pad(o['object_mask'],((0,0),(400,400),(500,500)))
    truth['intrinsics'][:,0,2]+=500;truth['intrinsics'][:,1,2]+=400
    shifted={k:v.copy() for k,v in truth.items()}
    shifted['object_translation'][0,0]+=.012
    exact=frame_overlap(shifted,o,0)
    cropped=frame_overlap(shifted,o,0,160,crop_to_object=True)
    legacy=frame_overlap(shifted,o,0,160)
    assert 0<exact<1
    assert exact==cropped
    assert abs(legacy-exact)>.005


def test_crop_never_hides_candidate_pixels_outside_target():
    truth=box_geometry(1);o=observations(truth)
    shifted={k:v.copy() for k,v in truth.items()}
    shifted['object_translation'][0,0]+=.12
    exact=frame_overlap(shifted,o,0)
    assert frame_overlap(shifted,o,0,160,crop_to_object=True)==exact
    assert exact<.6


def test_anchor_budget_covers_early_and_late_segments(make_bundle,monkeypatch):
    from real2sim2real.reconstruction_repair import repairs
    g=box_geometry(19);o=observations(g);o['reliable'][9]=False
    b=make_bundle(g=g,o=o)
    visited=[]
    monkeypatch.setattr(repairs,'frame_overlap',lambda *args,**kwargs:0.)
    def fit(bundle,parent,i,config,neighbors):
        visited.append(i)
        return np.eye(3),np.zeros(2)
    monkeypatch.setattr(repairs,'_pose_fit',fit)
    cfg=only('pose_recovery',anchor_spacing_s=.01,max_fit_anchors=2)
    p=pose_recovery_propose(b,g,cfg)
    assert len(visited)==2
    assert visited[0]<9<visited[1]
    for key in g:np.testing.assert_array_equal(p.geometry[key],g[key])


def test_abrupt_pose_corrections_are_limited_before_acceptance():
    g=box_geometry(21)
    t=np.linspace(0,2*np.pi,21)
    g['object_translation'][:,0]=.01*np.sin(t)
    g['object_rotation']=Rotation.from_rotvec(np.c_[np.zeros(21),np.zeros(21),.1*np.sin(t)]).as_matrix()
    bad={k:v.copy() for k,v in g.items()}
    bad['object_translation'][10,0]+=.05
    bad['object_rotation'][10]=Rotation.from_euler('z',20,degrees=True).as_matrix()@g['object_rotation'][10]
    cfg=only('pose_recovery',pose_backtracking_steps=12)
    candidate,fraction=limit_pose_motion(g,bad,cfg)
    assert 0<fraction<1
    a,b=motion_statistics(g),motion_statistics(candidate)
    for metric in ('center_acceleration_m_s2','angular_speed_deg_s'):
        for stat in ('p95','max'): assert b[metric][stat]<=max(a[metric][stat]*1.05,1e-6)
    np.testing.assert_array_equal(candidate['object_translation'][:,2],g['object_translation'][:,2])
    np.testing.assert_array_equal(candidate['object_rotation'][[0,-1]],g['object_rotation'][[0,-1]])
    assert np.allclose(np.linalg.det(candidate['object_rotation']),1)
    assert bad['object_translation'][10,0]>.04  # caller's proposal was not mutated


def test_motion_limiter_does_not_bypass_contact_guard(make_bundle):
    desired=box_geometry();parent={k:v.copy() for k,v in desired.items()}
    parent['object_translation'][:,0]=.04
    o=observations(desired);o['hand_points_world'][:]=[.20,0.,2.]
    b=make_bundle(g=parent,o=o);cfg=only('pose_recovery')
    candidate,fraction=limit_pose_motion(parent,desired,cfg)
    assert fraction==1.
    proposal=RepairProposal('pose_recovery',candidate,'Smooth but contact-breaking proposal')
    accepted=proposal_validate(b,parent,proposal,geometry_scores(b),geometry_scores(b,candidate),cfg)
    assert not accepted['accepted']
    assert 'hand_surface_separation_regressed' in accepted['reasons']


def test_motion_backtracking_falls_back_when_no_bounded_step_is_safe():
    g=box_geometry();bad={k:v.copy() for k,v in g.items()}
    bad['object_translation'][4,0]=.05
    candidate,fraction=limit_pose_motion(g,bad,only('pose_recovery'))
    assert fraction==0.
    for key in g:np.testing.assert_array_equal(candidate[key],g[key])
