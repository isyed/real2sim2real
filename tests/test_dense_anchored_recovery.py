import numpy as np
import pytest
from test_observation_preservation import fixture
from real2sim2real.reconstruction_repair.dense_anchored_recovery import initialize_fit_poses, refine
from real2sim2real.reconstruction_repair.temporal_pose import TemporalPoseConfig


def test_initialization_uses_fit_proposals_and_keeps_registered_anchors_exact():
    data, poses = fixture(); anchors = poses[[0, -1]].copy(); before = data['poses_camera'].copy()
    base, seeded, receipt = initialize_fit_poses(data, np.array([0, 4]), anchors, np.array([2]), poses[[2]])
    np.testing.assert_array_equal(seeded[[0, -1]], anchors)
    np.testing.assert_array_equal(seeded[2], poses[2])
    np.testing.assert_array_equal(data['poses_camera'], before)
    assert not base['locked'][2] and receipt['dense']['heldout_poses_used'] is False


def test_heldout_poses_cannot_become_initialization_knots():
    data, poses = fixture()
    with pytest.raises(ValueError, match='held-out'):
        initialize_fit_poses(data, np.array([0, 4]), poses[[0, 4]], np.array([1]), poses[[1]])


def test_duplicate_proposals_and_anchor_edits_are_rejected():
    data, poses = fixture()
    with pytest.raises(ValueError):
        initialize_fit_poses(data, np.array([0, 4]), poses[[0, 4]], np.array([2, 2]), poses[[2, 2]])
    bad = poses[[0]].copy(); bad[0, 0, 3] += .01
    with pytest.raises(ValueError, match='anchor'):
        initialize_fit_poses(data, np.array([0, 4]), poses[[0, 4]], np.array([0]), bad)


def test_dense_initialization_cannot_certify_heldout_depth_failure():
    data, poses = fixture(); data['depth_m'][1] += .4
    _, report = refine(data, np.array([0, 4]), poses[[0, 4]], np.array([2]), poses[[2]],
        config=TemporalPoseConfig(max_function_calls=1, max_seconds=2, benefit_mode='observation'))
    assert not report['decision']['accepted'] and 'depth_gate' in report['decision']['reasons']
    assert report['requires_gpu_validation'] and not report['full_reference_accepted']
    assert report['comparison_baseline']=='unchanged_registered_anchor_only_hypothesis'


def test_final_checks_keep_anchor_only_baseline_after_optimization(monkeypatch):
    from real2sim2real.reconstruction_repair import dense_anchored_recovery as module
    data, poses = fixture(); calls=[]
    def acceptance(window,before,after,original,candidate,config):
        calls.append(original.copy())
        return {'accepted':False,'reasons':['test_rejection']}
    monkeypatch.setattr(module,'acceptance',acceptance)
    monkeypatch.setattr(module,'refine_window',lambda window, config:(window['poses_camera'].copy(),{}))
    _,report=module.refine(data,np.array([0,4]),poses[[0,4]],np.array([2]),poses[[2]])
    assert len(calls)==2
    np.testing.assert_array_equal(calls[0],calls[1])
    assert not np.array_equal(calls[1][2],poses[2])
    assert not report['decision']['accepted']
