"""Checks for the independently installable translation backtracking API."""
import numpy as np
import pytest
from test_temporal_pose import fixture,CFG
from real2sim2real.reconstruction_repair.translation_backtracking import backtrack_translation


def test_smaller_update_preserves_rotations_anchors_and_noneditable_frames():
    data=fixture();original=data['poses_camera'].copy();proposal=original.copy();proposal[:,2,3]-=.02
    editable=np.zeros(21,bool);editable[8:13]=True
    candidate,report=backtrack_translation(data,proposal,CFG,fractions=(.5,),editable=editable)
    assert np.array_equal(candidate[~editable],original[~editable])
    assert np.array_equal(candidate[:,:3,:3],original[:,:3,:3])
    assert np.array_equal(data['poses_camera'],original)
    assert not report['approved_for_use']


def test_extra_interaction_rejection_cannot_be_overridden():
    data=fixture();proposal=data['poses_camera'].copy();proposal[:,2,3]=2.
    candidate,report=backtrack_translation(data,proposal,CFG,fractions=(.5,),extra_gate=lambda _:dict(passed=False))
    assert not report['accepted']
    assert np.array_equal(candidate,data['poses_camera'])
    assert 'extra_interaction_gate' in report['trials'][0]['decision']['reasons']


def test_rejects_rotation_changes_and_invalid_fraction_order():
    data=fixture();proposal=data['poses_camera'].copy();proposal[4,:3,:3]=np.diag([-1.,-1.,1.])
    with pytest.raises(ValueError,match='rotations'):backtrack_translation(data,proposal,CFG)
    with pytest.raises(ValueError,match='fractions'):backtrack_translation(data,data['poses_camera'],CFG,fractions=(.25,.5))
