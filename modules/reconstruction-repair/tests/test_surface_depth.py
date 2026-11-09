import numpy as np
import pytest
from real2sim2real.reconstruction_repair.surface_depth import front_surface_depth, compare_surface_depth


def test_perspective_correct_depth_and_batch_invariance():
    # Triangle projects to (0,0), (4,0), (0,4). At (1,1), inverse
    # depth is .5/1 + .25/2 + .25/4; linear Z would be incorrect.
    v = np.array([[0., 0., 1.], [8., 0., 2.], [0., 16., 4.]])
    f = np.array([[0, 1, 2]])
    a = front_surface_depth(v, f, np.eye(3), (6, 6), batch_samples=2)
    b = front_surface_depth(v, f[:, ::-1], np.eye(3), (6, 6))
    np.testing.assert_allclose(a, b)
    assert a[1, 1] == pytest.approx(1/.6875)
    assert np.isinf(a[5, 5])


def test_front_surface_wins_regardless_of_rear_vertex_density():
    front = np.array([[0., 0., 2.], [8., 0., 2.], [0., 8., 2.]])
    v = np.concatenate([front, front*2])
    f = np.array([[3, 4, 5], [0, 1, 2], [5, 4, 3]])
    result = front_surface_depth(v, f, np.eye(3), (5, 5))
    assert result[1, 1] == 2
    assert result[0, 4] == 2


def test_offscreen_and_degenerate_triangles():
    v = np.array([[-10., -10., 1.], [10., 0., 1.], [0., 10., 1.]])
    z = front_surface_depth(v, np.array([[0, 1, 2], [0, 0, 0]]), np.eye(3), (3, 3))
    np.testing.assert_allclose(z, 1)


def test_invalid_geometry_fails_closed():
    v = np.array([[0., 0., 1.], [1., 0., 1.], [0., 1., 1.]])
    with pytest.raises(ValueError, match='indices'):
        front_surface_depth(v, np.array([[0, 1, 9]]), np.eye(3), (3, 3))
    v[0, 2] = -1
    with pytest.raises(ValueError, match='near plane'):
        front_surface_depth(v, np.array([[0, 1, 2]]), np.eye(3), (3, 3))


def test_occluded_and_invalid_pixels_do_not_supply_evidence():
    surface=np.full((4,4),2.)
    predicted=surface.copy()
    predicted[0]=3.
    occ=np.zeros((4,4),bool);occ[1:]=True
    result=compare_surface_depth(surface,predicted,np.ones((4,4),bool),occluder_mask=occ,min_points=4)
    assert result['median_absolute_m']==1
    assert not result['consistency_passed']
    predicted[0]=[np.nan,0,np.inf,65.535]
    result=compare_surface_depth(surface,predicted,np.ones((4,4),bool),occluder_mask=occ,min_points=4)
    assert result['points']==0
    assert result['median_absolute_m'] is None
    assert not result['consistency_passed']


def test_erosion_receipt_and_agreement_never_approve_candidate():
    surface=np.full((5,5),2.)
    predicted=surface.copy();predicted[[0,-1]]=6.;predicted[:,[0,-1]]=6.
    result=compare_surface_depth(surface,predicted,np.ones((5,5),bool),erosion_pixels=1,min_points=9)
    assert result['points']==9
    assert result['consistency_passed']
    assert not result['candidate_approved']
    assert not result['metric_truth_validated']
    assert not compare_surface_depth(surface,predicted,np.ones((5,5),bool),erosion_pixels=1,min_points=10)['consistency_passed']


def test_comparison_rejects_mismatched_masks():
    with pytest.raises(ValueError,match='boolean'):
        compare_surface_depth(np.ones((3,3)),np.ones((3,3)),np.ones((3,3)))
