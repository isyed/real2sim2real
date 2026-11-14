import numpy as np
import pytest
from real2sim2real.retargeting.clearance_envelope import vertical_clearance_target
from scipy.spatial.transform import Rotation


def test_entire_rotated_body_clears_obstacles_with_reserved_tracking_error():
    points = np.array(
        [[x, y, z] for x in [-0.04, 0.04] for y in [-0.04, 0.04] for z in [-0.09, 0.09]]
    )
    r = vertical_clearance_target(
        points, 0.24, clearance_m=0.01, tilt_reserve_rad=0.35, tracking_reserve_m=0.03
    )
    assert r["required_center_z_m"] > 0.37
    assert not r["path_admitted"]
    rng = np.random.default_rng(14)
    for _ in range(100):
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        rotation = Rotation.from_rotvec(axis * rng.uniform(0, 0.35)).as_matrix()
        actual_z = (points @ rotation.T)[:, 2] + r["required_center_z_m"] - 0.03
        assert actual_z.min() >= 0.25


def test_zero_rotation_exact_vertex_bound_and_input_validation():
    r = vertical_clearance_target([[0, 0, -0.09], [0, 0, 0.09]], 0.2, clearance_m=0.01)
    assert r["required_center_z_m"] == pytest.approx(0.3)
    for points, kwargs in [
        ([], {}),
        ([[np.nan, 0, 0]], {}),
        ([[0, 0, 0]], {"tilt_reserve_rad": 4}),
        ([[0, 0, 0]], {"tracking_reserve_m": -0.01}),
    ]:
        with pytest.raises(ValueError):
            vertical_clearance_target(points, 0.2, clearance_m=0.01, **kwargs)
