import copy
import itertools
import numpy as np
import pytest
from real2sim2real.reconstruction_repair.joint_depth_pose import fit, refine, Config


def problem():
    vertices = np.array(list(itertools.product([-.1, .1], repeat=3)))
    times = np.array([0., .1, .2])
    seeds = np.tile(np.eye(4), (3, 1, 1)); seeds[:, 2, 3] = 3.
    k = np.array([[500., 0, 320], [0, 500., 240], [0, 0, 1]])
    theta = np.linspace(0, 2*np.pi, 16, endpoint=False)
    axes = np.c_[np.cos(theta), np.sin(theta)]
    camera = vertices + [0, 0, 3.15]; uv = camera @ k.T; uv = uv[:, :2] / uv[:, 2:]
    observations = [dict(K=k, directions=axes, outline_px=np.max(uv @ axes.T, axis=0),
        surface_points_local=vertices[vertices[:, 2] < 0], depth_predictions_m=np.array([3.035, 3.065]),
        depth_source_names=['full', 'crop'], support_plane_camera=np.array([0, -1., 0, 1]),
        source_provenance={'rgb_sha256': 'synthetic-source'}, ground_truth_used=False) for _ in times]
    return vertices, times, seeds, observations, np.linspace(0, .2, 5)


def test_joint_fit_moves_uncertain_anchor_without_changing_inputs_or_claiming_acceptance():
    args = problem(); original = copy.deepcopy(args)
    poses, receipt = fit(*args, config=Config(max_calls=600, max_seconds=15))
    assert abs(poses[0, 2, 3] - 3.15) < .04
    assert receipt['best_proxy_cost'] < receipt['initial_proxy_cost']
    assert not receipt['reference_approved'] and not receipt['heldout_observations_used']
    assert receipt['latent_depth_is_not_validation_evidence']
    assert np.array_equal(args[2], original[2])
    assert np.array_equal(args[3][0]['depth_predictions_m'], original[3][0]['depth_predictions_m'])
    assert np.allclose(poses[:, :3, :3] @ poses[:, :3, :3].swapaxes(1, 2), np.eye(3))


def test_heldout_queries_cannot_extrapolate_beyond_fit_interval():
    args = list(problem()); args[-1] = np.array([0., .3])
    with pytest.raises(ValueError): fit(*args)


@pytest.mark.parametrize('key,value', [('ground_truth_used', True),
    ('source_provenance', {}), ('depth_source_names', ['same', 'same']),
    ('depth_predictions_m', np.array([np.nan, 3.]))])
def test_missing_or_invalid_independent_source_evidence_rejected(key, value):
    args = problem(); args[3][0][key] = value
    with pytest.raises(ValueError): fit(*args)


def test_budget_is_reported_without_approving_partial_optimization():
    _, receipt = fit(*problem(), config=Config(max_calls=1))
    assert receipt['budget_reached'] and not receipt['reference_approved']


def test_rigid_pose_validation_and_frame_order():
    args = problem(); args[2][0, 0, 0] = 2
    with pytest.raises(ValueError): fit(*args)


def test_surface_refresh_only_passes_fit_knots_and_preserves_global_work_limit():
    v,t,seeds,observations,query=problem();calls=[]
    def builder(knots):
        calls.append(knots.copy());assert knots.shape==(3,4,4)
        return observations
    poses,receipt=refine(v,t,seeds,builder,query,config=Config(max_calls=700),max_rounds=2)
    assert len(calls)==2 and poses.shape==(5,4,4) and receipt['calls']<=700
    assert receipt['initial_seed_bounds_passed'] and not receipt['reference_approved']
