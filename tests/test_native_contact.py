import numpy as np
import trimesh
from real2sim2real.egocentric.native_contact import (
    side_coordinates,
    tangent_evidence,
    place_distal_sweep,
)
from real2sim2real.retargeting.pad_contacts import convex_outward_point


def test_fixed_taper_signed_residual_and_height():
    h, e = side_coordinates([[0.04, 0, 0.1], [0.043, 0, 0.1]], [0.03, 0.05, 0.2])
    assert np.allclose(h, 0.5) and abs(e[0]) < 1e-10 and e[1] > 0


def test_patch_coverage_and_unknown_are_distinct():
    x, y = np.meshgrid(np.linspace(-0.009, 0.009, 8), np.linspace(-0.009, 0.009, 8))
    p = np.c_[x.ravel(), y.ravel(), np.full(x.size, 0.002)]
    r = tangent_evidence(p, [0, 0, 0], [0, 0, 1], [1, 0, 0], [0.02, 0.02])
    assert (
        r["coverage_fraction"] == 1
        and abs(r["normal_offset_m"] - 0.002) < 1e-9
        and r["normal_error_deg"] < 1e-6
    )
    assert (
        tangent_evidence(p + [1, 0, 0], [0, 0, 0], [0, 0, 1], [1, 0, 0], [0.02, 0.02])[
            "plane_p90_m"
        ]
        is None
    )


def test_real_convex_surface_exit_and_sweep_orientation():
    hull = trimesh.creation.box(extents=[0.02, 0.01, 0.03])
    point, normal = convex_outward_point(hull, [0, 1, 0])
    assert np.allclose(point, [0, 0.005, 0]) and np.allclose(normal, [0, 1, 0])
    # local outward z points into the object; the body stays outside it.
    sweep = place_distal_sweep(
        [[0, 0, -0.01]],
        [0, 0, 0],
        np.eye(3),
        [1, 0, 0],
        [1, 0, 0],
        [0, 0, 1],
        [0, 0.05],
    )
    assert np.allclose(sweep[:, 0, 0], [1.01, 1.06])
