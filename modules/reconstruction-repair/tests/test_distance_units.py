"""Small triangles must remain finite surfaces across coordinate units."""
import numpy as np
import pytest
from real2sim2real.reconstruction_repair.geometry import points_mesh_distance


@pytest.mark.parametrize('scale', [1e-3, 1., 100.])
@pytest.mark.parametrize('dense', [False, True])
def test_small_triangle_distance_is_unit_invariant(scale, dense):
    vertices = np.array([[0.,0.,0.], [1e-4,0.,0.], [0.,1e-4,0.]])
    points = np.array([[.005,.005,.01]])
    faces = np.tile([[0,1,2]], (2050 if dense else 1, 1))
    expected = np.sqrt(2*(.005-.00005)**2 + .01**2)
    actual = points_mesh_distance(points*scale, vertices*scale, faces)/scale
    np.testing.assert_allclose(actual, [expected], atol=1e-12, rtol=1e-12)


def test_distance_preserves_rigid_motion_with_small_and_degenerate_triangles():
    vertices = np.array([[0.,0.,0.], [1e-4,0.,0.], [0.,1e-4,0.]])
    points = np.array([[.005,.005,.01], [2e-5,2e-5,.01]])
    faces = np.array([[0,1,2], [0,0,1]])
    rotation = np.array([[0.,0.,1.], [1.,0.,0.], [0.,1.,0.]])
    translation = np.array([1.,-2.,3.])
    expected = points_mesh_distance(points, vertices, faces)
    actual = points_mesh_distance(points@rotation.T+translation,
                                  vertices@rotation.T+translation, faces)
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)
