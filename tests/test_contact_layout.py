import numpy as np
import pytest
import trimesh

from real2sim2real.retargeting.contact_layout import (
    contact_targets,
    gravity_lever,
    surface_layouts,
)


def test_gravity_lever_ignores_height_and_matches_known_torque():
    row = gravity_lever([0.2, 0, 10], [0, 0, 0])
    assert row["gravity_lever_m"] == pytest.approx(0.2)
    np.testing.assert_allclose(row["gravity_torque_per_kg_nm"], [0, -1.962, 0])
    assert gravity_lever([0, 0, 10], [0, 0, 0])["gravity_lever_m"] == 0
    with pytest.raises(ValueError):
        gravity_lever([0, 0, 0], [0, 0, 0], [0, 0, 0])


def test_reversing_hand_swaps_thumb_and_finger_contact_surfaces():
    mesh = trimesh.creation.box(extents=[0.04, 0.2, 0.04])
    pair = dict(
        center_m=[0, 0, 0],
        width_m=0.04,
        close_axis=[1, 0, 0],
        along_axis=[0, 1, 0],
        endpoints_m=[[-0.02, 0, 0], [0.02, 0, 0]],
        targets_m=[[-0.02, 0, 0]],
    )
    forward = contact_targets(mesh, pair, [0.03, 0.01, -0.01, -0.03])
    reverse = contact_targets(mesh, pair, [-0.03, -0.01, 0.01, 0.03], direction=-1)
    np.testing.assert_allclose(forward[:, 0], [-0.02, 0.02, 0.02, 0.02, 0.02])
    np.testing.assert_allclose(reverse[:, 0], [0.02, -0.02, -0.02, -0.02, -0.02])
    np.testing.assert_allclose(reverse[1:, 1], [-0.03, -0.01, 0.01, 0.03])
    # Legacy outer-ray pairs lack explicit endpoints; their reversed target
    # construction must exchange surfaces too.
    legacy = {k: v for k, v in pair.items() if k != "endpoints_m"}
    np.testing.assert_allclose(
        contact_targets(mesh, legacy, [-0.03, -0.01, 0.01, 0.03], direction=-1), reverse
    )
    with pytest.raises(ValueError):
        contact_targets(mesh, pair, [0.2])
    mesh.apply_scale([1, 0.5, 1])
    assert contact_targets(mesh, pair, [0.09]) is None


def test_local_spans_find_vessel_wall_without_pairing_empty_cavity():
    mesh = trimesh.creation.annulus(r_min=0.08, r_max=0.11, height=0.06, sections=64)
    mesh.apply_translation([0, 0, 0.06])
    audit = {}
    rows = surface_layouts(
        mesh,
        min_width_m=0.02,
        max_width_m=0.05,
        minimum_height_m=0.045,
        max_candidates=6,
        finger_offsets_m=[-0.015, 0.015],
        yaw_samples=4,
        diagnostics=audit,
    )
    assert rows
    assert audit["sampled_solid_spans"] >= audit["width_admitted_spans"] >= len(rows)
    assert len(audit["solid_span_width_m"]) == audit["sampled_solid_spans"]
    assert audit["selected_pairs"] == len(rows)
    for row in rows:
        assert 0.02 <= row["width_m"] <= 0.05
        midpoint = np.asarray(row["endpoints_m"]).mean(0)
        assert 0.079 < np.linalg.norm(midpoint[:2]) < 0.111
        # Rays near the centre cannot incorrectly bridge the empty interior.
        for key in ("targets_m", "reverse_targets_m"):
            targets = np.asarray(row[key])
            distances = trimesh.proximity.closest_point(mesh, targets)[1]
            assert distances.max() < 1e-8


def test_local_spans_reject_open_geometry_and_invalid_search():
    mesh = trimesh.creation.box(extents=[0.04, 0.2, 0.04])
    args = dict(min_width_m=0.02, max_width_m=0.06, minimum_height_m=-0.01)
    with pytest.raises(ValueError):
        surface_layouts(mesh, **args, yaw_samples=100)
    with pytest.raises(ValueError):
        surface_layouts(mesh, **args, gravity_lever_weight=float("nan"))
    mesh.update_faces(np.arange(len(mesh.faces) - 1))
    with pytest.raises(ValueError, match="closed"):
        surface_layouts(mesh, **args)


def test_fixed_validation_allows_longer_holds_but_rejects_retuning():
    from copy import deepcopy
    from real2sim2real.controller_actor_training.object_grasp_pilot import (
        verify_frozen_controller,
    )

    config = dict(
        rollout=dict(approach_s=1.0, lift_s=2.0, hold_s=3.0, close_offset_rad=0.15),
        orientation_feedback=dict(gain_s_inv=3.0),
    )
    selected = dict(
        configuration=deepcopy(config),
        controller=dict(
            orientation_feedback=dict(gain_s_inv=3.0), close_offset_rad=0.45
        ),
    )
    longer = deepcopy(config)
    longer["rollout"]["hold_s"] = 10.0
    verify_frozen_controller(longer, selected)
    for key, value in [("lift_s", 1.0), ("hold_s", 1.0)]:
        changed = deepcopy(config)
        changed["rollout"][key] = value
        with pytest.raises(ValueError):
            verify_frozen_controller(changed, selected)
    changed = deepcopy(config)
    changed["orientation_feedback"]["gain_s_inv"] = 4.0
    with pytest.raises(ValueError, match="feedback"):
        verify_frozen_controller(changed, selected)
