import numpy as np
import pytest
from real2sim2real.controller_actor_training.reach_audit import summarize_contacts


def test_incidental_contact_and_failed_state_contact_do_not_count_as_valid_phase_contact():
    # Nontrivial slot order: hand 1, hand 0, hand 1.
    counts=np.array([[2,0,1],[0,1,0],[0,0,4],[1,0,0]])
    phases=np.array([[False,False],[True,False],[False,True],[False,True]])
    result=summarize_contacts(counts,np.array([1,0,1]),phases,valid_frames=np.array([True,True,False,True]))
    left,right=result['hands']
    assert left['valid_phase_contact_frames']==1
    assert right['physical_contact_frames']==3
    assert right['valid_contact_frames']==2
    assert right['valid_phase_contact_frames']==1
    assert right['valid_outside_phase_contact_frames']==1
    assert result['grasp_success_established'] is False


def test_reject_unlabelled_or_fractional_contacts():
    with pytest.raises(ValueError):
        summarize_contacts(np.array([[.5]]),np.array([0]),np.array([[True]]))
    with pytest.raises(ValueError):
        summarize_contacts(np.array([[1]]),np.array([2]),np.array([[True]]))
