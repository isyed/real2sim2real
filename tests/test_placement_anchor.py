import numpy as np
from real2sim2real.egocentric.container_axis import endpoints
from real2sim2real.egocentric.placement_anchor import fit_anchor
from real2sim2real.egocentric.tapered_container import surface


def test_occluded_endpoint_cannot_move_static_supported_anchor():
    dims = [0.03, 0.04, 0.17]
    K = np.array([[600.0, 0, 320], [0, 600, 240], [0, 0, 1]])
    T = np.eye(4)
    T[:3, 3] = [0.04, 0.02, 0]
    views = []
    for x in [0, 0.02, -0.02]:
        C = np.eye(4)
        C[:3, 3] = [x, 0, -0.8]
        N = np.linalg.inv(C) @ T
        p = surface(*dims)[::30] @ N[:3, :3].T + N[:3, 3]
        uv = endpoints(N, dims[2], K)
        uv[0] += 100
        views.append(
            {
                "points_camera_m": p,
                "endpoint_pixels": uv,
                "endpoint_visible": [False, True],
                "world_from_camera": C,
                "intrinsics": K,
            }
        )
    guess = T.copy()
    guess[0, 3] += 0.01
    a, _ = fit_anchor(views, dims, guess, upright_on_table_z_m=0)
    for v in views:
        v["endpoint_pixels"][0] += 1000
    b, _ = fit_anchor(views, dims, guess, upright_on_table_z_m=0)
    np.testing.assert_allclose(a, b, atol=1e-10)
    np.testing.assert_allclose(a, T, atol=1e-5)
