import json

import numpy as np
import pytest

from real2sim2real.artifacts import digest
from real2sim2real.egocentric.hand_depth_support import (
    prepare_boundaries,
    signed_support_distance,
    support_region,
    load_boundaries,
)
from real2sim2real.egocentric.hand_observation_fit import measured_surface
from real2sim2real.egocentric.hand_sequence import select_components, sequence_options
from real2sim2real.egocentric.point_hand_experiment import runtime_config


def detections():
    landmarks = np.full((3, 2, 21, 2), [50, 30], dtype=float)
    landmarks[:, :, 0] = [50, 80]
    return landmarks, np.ones((3, 2), bool), np.array([10, 11, 12])


def test_withheld_detections_cannot_change_boundary():
    landmarks, accepted, frames = detections()
    before = prepare_boundaries(landmarks, accepted, frames, [11])
    landmarks[1] = 0
    accepted[1] = False
    after = prepare_boundaries(landmarks, accepted, frames, [11])
    for key in before:
        np.testing.assert_equal(before[key], after[key])
    assert not before["available"][1].any()
    assert support_region((100, 100), before, 1, 0, 10).all()


def test_boundary_keeps_hand_and_margin_without_depth_or_pose():
    boundary = prepare_boundaries(*detections())
    region = support_region((100, 100), boundary, 0, 0, 10)
    assert region[:91].all()
    assert not region[91:].any()
    landmarks, accepted, frames = detections()
    landmarks[0, 0] = np.nan
    accepted[2, 1] = False
    missing = prepare_boundaries(landmarks, accepted, frames)
    assert not missing["available"][0, 0]
    assert not missing["available"][2, 1]


@pytest.mark.parametrize("margin", [True, -1, 101, np.nan, np.inf])
def test_invalid_margin(margin):
    with pytest.raises(ValueError):
        signed_support_distance([[1, 2]], [0, 0], [0, 1], margin)


def test_unit_direction_required():
    with pytest.raises(ValueError, match="unit palm"):
        signed_support_distance([[1, 2]], [0, 0], [0, 2], 10)


def test_receipt_blocks_changed_timeline_labels_and_content(tmp_path):
    _, _, frames = detections()
    boundary = prepare_boundaries(*detections(), [11])
    times = np.arange(3) * 0.05
    path = tmp_path / "boundaries.npz"
    np.savez(path, **boundary, frame_ids=frames, timestamps_s=times)
    receipt = dict(
        schema="real2sim2real.hand-depth-support/1",
        boundaries_sha256=digest(path),
        blocked_frame_ids=[11],
    )
    (tmp_path / "receipt.json").write_text(json.dumps(receipt))
    loaded, _ = load_boundaries(tmp_path, frames, times, [11])
    np.testing.assert_equal(loaded["available"], boundary["available"])
    with pytest.raises(ValueError, match="held-out"):
        load_boundaries(tmp_path, frames, times, [])
    with pytest.raises(ValueError, match="timeline"):
        load_boundaries(tmp_path, frames, times + 1, [11])
    boundary["available"][1] = True
    np.savez(path, **boundary, frame_ids=frames, timestamps_s=times)
    with pytest.raises(ValueError, match="receipt"):
        load_boundaries(tmp_path, frames, times, [11])
    receipt["boundaries_sha256"] = digest(path)
    (tmp_path / "receipt.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="Withheld"):
        load_boundaries(tmp_path, frames, times, [11])


def test_reserved_depth_and_excluded_forearm_do_not_change_fit():
    y, x = np.indices((60, 60))
    depth, mask, support = np.full((60, 60), 0.6), np.ones((60, 60), bool), y < 40
    k = np.array([[100, 0, 30], [0, 100, 30], [0, 0, 1]])
    args = (mask, [0, 0, 60, 60], k)
    before, reserved_before = measured_surface(depth, *args, support_mask=support)
    depth[(x // 3 + y // 3) % 2 == 1] += 0.05
    depth[~support] = 1.4
    after, reserved_after = measured_surface(depth, *args, support_mask=support)
    np.testing.assert_array_equal(before, after)
    assert not np.array_equal(reserved_before, reserved_after)
    assert (before[:, 1] < 0.06).all()
    with pytest.raises(ValueError, match="Boolean"):
        measured_surface(depth, *args, support_mask=support.astype(float))


def test_all_supported_fitting_still_splits_missing_and_time_gaps():
    observed = np.ones((8, 2), bool)
    observed[3, 0] = False
    times = np.arange(8) * 0.05
    times[6:] += 1
    selected, groups = select_components(observed, observed, times)
    np.testing.assert_array_equal(selected, observed)
    assert len(groups) == 5
    for group in groups:
        np.testing.assert_array_equal(group["selected"], group["window"])
    with pytest.raises(ValueError, match="boolean"):
        sequence_options(dict(fit_all_supported=1))


def test_only_prepared_boundary_folder_is_mounted(tmp_path):
    config = dict(
        output=str(tmp_path / "run"),
        parent="/parent",
        reconstruction="/capture",
        frame_start=0,
        frame_count=3,
        heldout_frame_ids=[1],
        mask_ids=[2, 3],
        object_id=1,
        weights=dict(tapir="/tapir", hamer="/hamer"),
        hand_depth_support=dict(path="/prepared-boundaries", margin_px=30),
    )
    runtime, _ = runtime_config(config, "sequence")
    assert "/prepared-boundaries" in runtime["read_only_inputs"]
    assert (
        runtime["model_options"]["hand_depth_support"] == config["hand_depth_support"]
    )
    assert all(
        "review" not in p and "evaluation" not in p for p in runtime["read_only_inputs"]
    )
