import numpy as np
import pytest
import trimesh

from real2sim2real.egocentric.contact_observations import (
    observe_contacts,
    summarize_contact,
)
from real2sim2real.retargeting.contact_proxy import local_span_audit, merge_intervals


def observation_inputs():
    return dict(
        points_object=[[0, 0, 0]],
        normals_object=[[0, 0, -1]],
        camera_from_object=np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1.0]]
        ),
        intrinsics=np.array([[100, 0, 10], [0, 100, 10], [0, 0, 1.0]]),
        depth=np.full((21, 21), 1.002),
        object_mask=np.ones((21, 21), bool),
        occluder_mask=np.zeros((21, 21), bool),
    )


def test_contact_projection_uses_camera_transform_and_masks_foreground():
    args = observation_inputs()
    row = observe_contacts(**args)[0]
    assert row["observed"] and row["pixel"] == [10, 10]
    assert row["residual_m"] == pytest.approx(0.002)
    args["camera_from_object"][0, 3] = 0.02
    assert observe_contacts(**args)[0]["pixel"] == [12, 10]
    args["occluder_mask"][10, 12] = True
    assert observe_contacts(**args)[0]["reason"] == "masked_occluder"
    args["occluder_mask"][:] = False
    args["depth"][:] = np.nan
    assert observe_contacts(**args)[0]["reason"] == "insufficient_depth"


def test_contact_evidence_never_treats_missing_or_backfacing_as_success():
    args = observation_inputs()
    good = observe_contacts(**args)[0]
    assert summarize_contact([good])["evidence"] == "insufficient"
    assert summarize_contact([good] * 3)["evidence"] == "supported"
    args["depth"][:] = 1.02
    bad = observe_contacts(**args)[0]
    assert summarize_contact([bad] * 3)["evidence"] == "inconsistent"
    args["normals_object"] = [[0, 0, 1]]
    hidden = observe_contacts(**args)[0]
    assert hidden["reason"] == "backfacing_or_grazing"
    assert summarize_contact([hidden] * 20)["evidence"] == "insufficient"
    with pytest.raises(ValueError):
        summarize_contact([dict(observed=True, residual_m=float("nan"))] * 3)


def test_neighborhood_excludes_occluder_depth_and_outside_mask_pixels():
    args = observation_inputs()
    args["depth"][8:10] = 0.4
    args["occluder_mask"][8:10] = True
    args["depth"][11:13] = 5
    args["object_mask"][11:13] = False
    row = observe_contacts(**args)[0]
    assert row["observed"] and row["depth_pixels"] == 5
    assert row["residual_m"] == pytest.approx(0.002)
    args["camera_from_object"][2, 3] = -1
    assert observe_contacts(**args)[0]["reason"] == "behind_camera"


def test_proxy_audit_uses_union_not_internal_piece_faces():
    source = trimesh.creation.box(extents=[0.004, 0.1, 0.1])
    a = trimesh.creation.box(extents=[0.003, 0.1, 0.1])
    a.apply_translation([-0.0005, 0, 0])
    b = a.copy()
    b.apply_translation([0.001, 0, 0])
    row = local_span_audit(source, [a, b], [0, 0, 0], [1, 0, 0])
    assert row["accepted"]
    assert row["proxy_width_m"] == pytest.approx(0.004)
    assert row["boundary_error_m"] < 1e-9
    assert len(row["proxy_union_intervals_m"]) == 1
    assert merge_intervals([[0, 2], [1, 3], [4, 5]]) == [[0, 3], [4, 5]]


def test_thin_wall_relative_boundary_gate_rejects_global_tolerance_pass():
    source = trimesh.creation.box(extents=[0.003, 0.1, 0.1])
    proxy = trimesh.creation.box(extents=[0.005, 0.1, 0.1])
    row = local_span_audit(source, [proxy], [0, 0, 0], [1, 0, 0])
    assert not row["accepted"]
    assert row["boundary_error_m"] == pytest.approx(0.001)
    assert row["boundary_limit_m"] == pytest.approx(0.00075)
    reverse = local_span_audit(source, [proxy], [0, 0, 0], [-1, 0, 0])
    assert reverse["boundary_error_m"] == pytest.approx(row["boundary_error_m"])
    assert not local_span_audit(source, [proxy], [1, 0, 0], [0, 1, 0])["accepted"]
