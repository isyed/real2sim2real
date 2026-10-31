import numpy as np
import pytest
from real2sim2real.controller_actor_training.object_stability import settling_summary


def test_early_settling_is_separate_from_late_drift():
    q=np.zeros((151,7));q[:,6]=1.;q[:30,2]=np.linspace(.03,0,30)
    s=settling_summary(q)
    assert s['observed_tail_seconds']==1.
    assert s['displacement_m']==0.
    assert s['rotation_deg']==0.
    assert s['initial_pose_quality_established'] is False


def test_continued_spin_is_detected_and_bad_quaternions_rejected():
    q=np.zeros((151,7));angles=np.linspace(0,.3,151)
    q[:,5]=np.sin(angles/2);q[:,6]=np.cos(angles/2)
    assert settling_summary(q)['rotation_deg']>5.
    q[0,3:]=0
    with pytest.raises(ValueError):settling_summary(q)
