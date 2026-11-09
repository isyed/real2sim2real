import numpy as np
import pytest

torch = pytest.importorskip("torch")

from real2sim2real.egocentric.mano_refinement import bounded_vectors, rotation_matrices


def test_bounded_rotations_are_proper_and_differentiable_at_identity():
    x = torch.zeros((2, 3), requires_grad=True)
    delta = bounded_vectors(x, 0.5)
    r = rotation_matrices(delta)
    torch.testing.assert_close(r, torch.eye(3).expand(2, 3, 3))
    r[0, 1, 2].backward()
    assert torch.isfinite(x.grad).all()
    assert abs(float(x.grad[0, 0])) > 0.1
    large = bounded_vectors(torch.tensor([[100.0, -50.0, 300.0]]), 0.5)
    assert float(torch.linalg.vector_norm(large)) <= 0.500001
    matrices = rotation_matrices(large).detach().numpy()
    np.testing.assert_allclose(np.linalg.det(matrices), 1.0, atol=1e-6)
    np.testing.assert_allclose(
        matrices @ matrices.transpose(0, 2, 1), np.eye(3)[None], atol=1e-6
    )
