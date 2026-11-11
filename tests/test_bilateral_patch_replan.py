from types import SimpleNamespace

import numpy as np
import pytest

from real2sim2real.controller_actor_training.bilateral_patch_replan import finger_bounds
from real2sim2real.controller_actor_training.bilateral_patch_replan import (
    shifted_contacts,
    posture_seed,
)


def test_patch_posture_bounds_preserve_wrists_and_unselected_fingers():
    model = SimpleNamespace(
        jnt_range=np.array([[-0.3, 0.3], [-1, 1], [0, 1.5], [-1, 1]])
    )
    names = ["fixture_x", "fixture_rx", "thumb_IP", "index_DIP"]
    bounds = finger_bounds(
        model,
        dict(joints=np.arange(4)),
        names,
        [0, 0, 0, 0.3],
        ["thumb_IP"],
        0.25,
        "fixture_",
    )
    assert bounds["thumb_IP"] == [0, 0.25]
    for name in ["fixture_x", "fixture_rx", "index_DIP"]:
        assert bounds[name][1] - bounds[name][0] <= 2e-7 + 1e-10
    with pytest.raises(ValueError):
        finger_bounds(
            model,
            dict(joints=np.arange(4)),
            names,
            [0, 0, 0, 0.3],
            ["fixture_x"],
            0.25,
            "fixture_",
        )
    with pytest.raises(ValueError):
        finger_bounds(
            model,
            dict(joints=np.arange(4)),
            names,
            [0, 0, float("nan"), 0.3],
            ["thumb_IP"],
            0.25,
            "fixture_",
        )


def test_contact_projection_preserves_thin_wall_side_and_source_geometry():
    import trimesh

    mesh = trimesh.creation.box(extents=[0.001, 0.02, 0.02])
    vertices = mesh.vertices.copy()
    targets = np.array([[0.0005, 0, 0]])
    normals = np.array([[1.0, 0, 0]])
    points, new_normals = shifted_contacts(
        mesh,
        targets,
        normals,
        [dict(geom="index_pad")],
        dict(index=[-0.0007, 0.001, 0]),
    )
    assert points[0, 0] == pytest.approx(0.0005)
    assert new_normals[0, 0] > 0.99
    np.testing.assert_array_equal(mesh.vertices, vertices)
    np.testing.assert_array_equal(targets, [[0.0005, 0, 0]])
    with pytest.raises(ValueError):
        shifted_contacts(
            mesh, targets, normals, [dict(geom="index_pad")], dict(index=[0.01, 0, 0])
        )


def test_posture_seed_cannot_change_a_frozen_joint_or_expand_bounds():
    names = ["fixture_x", "index_PIP", "ring_DIP"]
    bounds = dict(fixture_x=[-1e-7, 1e-7], index_PIP=[0, 0.2], ring_DIP=[0, 0.4])
    result = posture_seed(names, [0, 0.15, 0.1], dict(index_PIP=0.1), bounds)
    np.testing.assert_allclose(result, [0, 0.2, 0.1])
    with pytest.raises(ValueError):
        posture_seed(names, [0, 0.15, 0.1], dict(fixture_x=0.1), bounds)
    with pytest.raises(ValueError):
        posture_seed(names, [0, 0.15, 0.1], dict(index_PIP=0.3), bounds)
