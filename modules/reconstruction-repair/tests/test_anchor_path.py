import numpy as np
import pytest
from real2sim2real.reconstruction_repair.anchor_path import select


def row(angle,error,admissible=True):
    from scipy.spatial.transform import Rotation
    p=np.eye(4);p[:3,:3]=Rotation.from_euler('z',angle,degrees=True).as_matrix()
    return dict(pose_camera=p,observation_error=error,admissible=admissible)


def test_avoids_independent_silhouette_flips_without_approving_trajectory():
    groups=[[row(0,.1)],[row(180,.01),row(2,.1)],[row(3,.1)]]
    poses,receipt=select([0,1,2],groups)
    assert receipt['selected_candidate_indices']==[0,1,0]
    assert receipt['requires_independent_validation'] and not receipt['reference_accepted']
    assert not receipt['hidden_motion_established'] and not receipt['admissibility_overridden']
    assert np.allclose(poses[0],np.eye(4))


def test_prior_can_be_disabled_and_never_rescues_a_rejected_hypothesis():
    _,receipt=select([0,1],[[row(0,.1)],[row(180,.01),row(2,.1)]],angular_rate_cost=0.)
    assert receipt['selected_candidate_indices']==[0,0]
    with pytest.raises(ValueError,match='admissible'):
        select([0,1],[[row(0,.1)],[row(0,.1,False)]])
