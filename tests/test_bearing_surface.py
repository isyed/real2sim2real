import numpy as np
from real2sim2real.egocentric.bearing_surface import bearing_skin


def test_masked_gap_cannot_become_a_bearing_bridge():
    k = np.array([[100.0, 0, 50], [0, 100, 50], [0, 0, 1]])
    x, y = np.meshgrid(np.arange(15, 86, 2), np.arange(25, 76, 2))
    points = np.c_[(x.ravel() - 50) / 100, (y.ravel() - 50) / 100, np.ones(x.size)]
    mask = np.ones((100, 100), bool)
    mask[:, 49:52] = False
    skin = bearing_skin(
        points, np.eye(4), k, mask, maximum_pixel_edge=12, maximum_metric_edge_m=0.15
    )
    assert skin["components"] == 2
    triangles = skin["pixels"][skin["faces"]]
    assert not np.any(
        (triangles[:, :, 0].min(1) < 49) & (triangles[:, :, 0].max(1) >= 52)
    )
    assert skin["thickness_m"] is None


def test_metric_depth_jump_remains_disconnected():
    k = np.array([[100.0, 0, 50], [0, 100, 50], [0, 0, 1]])
    x, y = np.meshgrid(np.arange(30, 71, 2), np.arange(30, 71, 2))
    z = np.where(x.ravel() < 50, 1.0, 1.3)
    points = np.c_[(x.ravel() - 50) / 100 * z, (y.ravel() - 50) / 100 * z, z]
    skin = bearing_skin(
        points, np.eye(4), k, np.ones((100, 100), bool), maximum_metric_edge_m=0.05
    )
    assert skin["components"] == 2
