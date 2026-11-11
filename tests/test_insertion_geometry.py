import numpy as np
import pytest
import trimesh

from real2sim2real.retargeting.insertion_geometry import (
    vertical_grid,
    union_columns,
    sweep_clearance,
    signed_clearance,
)


def test_opening_discovery_and_straight_probe_clearance():
    ring = trimesh.creation.annulus(r_min=0.01, r_max=0.02, height=0.006)
    grid = vertical_grid(ring, pitch_m=0.001)
    assert len(grid["openings"]) == 1
    center = grid["openings"][0]["center_xy_m"]
    assert np.linalg.norm(center) < 0.002
    line = [[0, 0, 0.03], [0, 0, -0.02]]
    good = sweep_clearance(
        [("ring", ring, None)], [line], radius_m=0.006, margin_m=0.001
    )
    assert good["accepted"] and not good["whole_robot_validated"]
    assert not sweep_clearance(
        [("ring", ring, None)], [line], radius_m=0.012, margin_m=0.001
    )["accepted"]


def test_union_membership_does_not_cancel_overlapping_solids():
    a = trimesh.creation.box(extents=[0.02, 0.02, 0.02])
    b = a.copy()
    b.apply_translation([0.005, 0, 0])
    union = trimesh.util.concatenate([a, b])
    assert signed_clearance(union, [[0, 0, 0]], convex_pieces=[a, b])[0] < 0
    np.testing.assert_array_equal(
        union_columns([a, b], [[0, 0], [0.1, 0]]), [True, False]
    )


def test_table_and_between_sample_obstacle_cannot_be_ignored():
    wall = trimesh.creation.box(extents=[0.001, 0.02, 0.02])
    wall.apply_translation([0.01, 0, 0])
    result = sweep_clearance(
        [("wall", wall, None)],
        [[[0, 0, 0], [0.02, 0, 0]]],
        radius_m=0.0001,
        margin_m=0,
        maximum_step_m=0.02,
    )
    assert not result["accepted"]  # endpoints alone would miss the wall
    table = trimesh.creation.box(extents=[0.2, 0.2, 0.02])
    table.apply_translation([0, 0, -0.01])
    result = sweep_clearance(
        [("table", table, None)],
        [[[0, 0, 0.02], [0, 0, 0.003]]],
        radius_m=0.004,
        margin_m=0.001,
    )
    assert not result["accepted"]
    with pytest.raises(ValueError):
        vertical_grid(wall, pitch_m=0.000001)
