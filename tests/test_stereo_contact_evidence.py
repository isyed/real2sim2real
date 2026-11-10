import numpy as np
import pytest

from real2sim2real.egocentric.stereo_contact_evidence import (
    match_rectified,
    triangulate_match,
)


def stereo_pair():
    left = np.random.default_rng(42).integers(0, 255, (60, 180), dtype=np.uint8)
    right = np.zeros_like(left)
    right[:, :-12] = left[:, 12:]
    return left, right


def test_match_recovers_depth_without_predicted_disparity_and_roundtrip():
    left, right = stereo_pair()
    row = match_rectified(
        left, right, [100, 30], maximum_disparity=40, vertical_radius=0
    )
    assert row["accepted"] and row["disparity_px"] == pytest.approx(12, abs=0.1)
    assert row["roundtrip_px"] < 0.1
    p = np.array([[120, 0, 0, 0], [0, 120, 0, 0], [0, 0, 1, 0]], float)
    q = p.copy()
    q[0, 3] = -12
    xyz = triangulate_match(row, [p, q])
    assert xyz[2] == pytest.approx(1, abs=0.01)


def test_missing_texture_repetitions_and_boundary_do_not_become_depth():
    flat = np.full((60, 180), 100, dtype=np.uint8)
    assert match_rectified(flat, flat, [100, 30])["reason"] == "low_texture"
    stripe = np.tile(np.tile([0, 0, 255, 255], 45), (60, 1)).astype(np.uint8)
    assert not match_rectified(stripe, stripe, [100, 30], maximum_disparity=40)[
        "accepted"
    ]
    left, right = stereo_pair()
    boundary = match_rectified(
        left, right, [100, 30], minimum_disparity=12, maximum_disparity=40
    )
    assert boundary["reason"] == "search_boundary"
    with pytest.raises(ValueError, match="Rejected"):
        triangulate_match(boundary, [np.eye(3, 4)] * 2)
    assert match_rectified(left, right, [0, 0])["reason"] == "outside_image"


def test_reverse_ambiguity_rejects_forward_unique_match():
    left, right = stereo_pair()
    left[27:34, 117:124] = left[27:34, 97:104]
    row = match_rectified(
        left, right, [100, 30], maximum_disparity=40, vertical_radius=0
    )
    assert not row["accepted"] and row["reason"] == "reverse_ambiguous_match"


def test_invalid_unbounded_inputs_rejected():
    left, right = stereo_pair()
    for options in [
        dict(radius=0),
        dict(maximum_disparity=10000),
        dict(minimum_uniqueness=0),
        dict(minimum_std=float("nan")),
    ]:
        with pytest.raises(ValueError):
            match_rectified(left, right, [100, 30], **options)


def test_stereo_surface_visibility_rejects_backfaces_and_hidden_surface():
    import trimesh
    from real2sim2real.controller_actor_training.stereo_contact_audit import (
        visible_surface,
    )

    mesh = trimesh.creation.box()
    mesh.apply_translation([0, 0, 1])
    points = np.array([[0, 0, 0.5], [0, 0, 1.5], [0.1, 0, 0.5]])
    normals = np.array([[0, 0, -1], [0, 0, -1], [0, 0, 1]])
    np.testing.assert_array_equal(
        visible_surface(mesh, points, normals, np.eye(4), 0.001), [True, False, False]
    )
