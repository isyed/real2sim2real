import numpy as np
import pytest
from real2sim2real.retargeting.container_colliders import tapered_shell
from scipy.spatial import ConvexHull


def test_shell_opening_is_empty_and_base_closed():
    pieces = tapered_shell(0.035, 0.043, 0.17, 0.002, 0.002)

    def inside(point):
        return any(
            np.all(
                ConvexHull(p["vertices_m"]).equations[:, :3] @ point
                + ConvexHull(p["vertices_m"]).equations[:, 3]
                <= 1e-10
            )
            for p in pieces
        )

    assert len(pieces) == 49 and not inside(np.array([0, 0, 0.169]))
    assert not inside(np.array([0, 0, 0.085])) and inside(np.array([0, 0, 0.001]))
    assert inside(np.array([0.038, 0, 0.085]))
    for p in pieces:
        assert p["volume_m3"] > 0
        triangles = p["vertices_m"][p["faces"]]
        signed_volume = np.einsum(
            "ij,ij->i", triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2])
        ).sum() / 6
        assert signed_volume == pytest.approx(p["volume_m3"])
    with pytest.raises(ValueError):
        tapered_shell(0.035, 0.043, 0.17, 0.03, 0.002)
