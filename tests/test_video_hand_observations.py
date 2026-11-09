import json

import numpy as np
import pytest
from PIL import Image

from real2sim2real.egocentric.hand_tracks import select_crops, validate_tracks
from real2sim2real.egocentric.hawor_worker import mirrored_camera, consecutive_windows
from real2sim2real.egocentric.hand_observation_fit import (
    measured_surface,
    visible_landmarks,
)
from real2sim2real.egocentric.hand_experiment import subset_archive
from real2sim2real.egocentric.interaction_io import Reconstruction


def test_crops_exist_without_mediapipe_and_keep_physical_hand_identity():
    masks = np.zeros((2, 80, 100), bool)
    masks[0, 10:40, 5:30] = True
    masks[1, 15:55, 65:95] = True
    boxes, origins, scores = select_crops([[60, 10, 100, 60]], [0.9], masks)
    np.testing.assert_array_equal(origins, ["mask", "detector"])
    np.testing.assert_array_equal(boxes[0], [5, 10, 30, 40])
    np.testing.assert_array_equal(boxes[1], [60, 10, 100, 60])
    assert scores[0] == 0 and scores[1] > 0


def test_ambiguous_detector_box_is_not_assigned_to_two_hands():
    masks = np.zeros((2, 50, 50), bool)
    masks[0, 10:30, 0:20] = True
    masks[1, 10:30, 30:50] = True
    _, origins, _ = select_crops([[0, 0, 50, 50]], [1], masks)
    assert origins.tolist() == ["mask", "mask"]
    boxes, origins, _ = select_crops([], [], np.zeros_like(masks))
    assert np.isnan(boxes).all() and origins.tolist() == ["missing", "missing"]


def test_tracks_reject_mismatched_timeline_and_false_provenance():
    data = dict(
        frame_ids=np.array([40, 41]),
        timestamps_s=np.array([2.0, 2.05]),
        boxes_xyxy=np.tile([1, 2, 20, 30], (2, 2, 1)),
        crop_valid=np.ones((2, 2), bool),
        crop_source=np.full((2, 2), "detector"),
    )
    validate_tracks(data, [40, 41], [2.0, 2.05])
    with pytest.raises(ValueError, match="window"):
        validate_tracks(data, [0, 1], [2.0, 2.05])
    data["crop_valid"][0, 1] = False
    with pytest.raises(ValueError, match="provenance"):
        validate_tracks(data, [40, 41], [2.0, 2.05])


def test_left_mirror_uses_actual_principal_point_and_is_involutive():
    center, principal = mirrored_camera([200, 300], [497, 383], 1014, 0)
    np.testing.assert_array_equal(center, [813, 300])
    np.testing.assert_array_equal(principal, [516, 383])
    a, b = mirrored_camera(center, principal, 1014, 0)
    np.testing.assert_array_equal(a, [200, 300])
    np.testing.assert_array_equal(b, [497, 383])
    assert center[0] - principal[0] == -(200 - 497)


def test_temporal_windows_never_bridge_missing_crops_or_publish_padding():
    valid = np.ones(23, bool)
    valid[8:11] = False
    windows = list(consecutive_windows(valid))
    assert [len(a) for a, _ in windows] == [8, 12]
    assert all(len(b) == 16 for _, b in windows)
    np.testing.assert_array_equal(
        np.concatenate([a for a, _ in windows]), np.flatnonzero(valid)
    )
    assert list(consecutive_windows(np.zeros(5, bool))) == []


def test_fit_surface_selection_never_uses_heldout_depth():
    depth = np.full((40, 40), 0.5)
    mask = np.ones_like(depth, bool)
    k = np.array([[100, 0, 20], [0, 100, 20], [0, 0, 1]])
    before, _ = measured_surface(depth, mask, [0, 0, 40, 40], k, max_points=100)
    y, x = np.indices(depth.shape)
    depth[(x // 3 + y // 3) % 2 == 1] = 1.3
    after, _ = measured_surface(depth, mask, [0, 0, 40, 40], k, max_points=100)
    np.testing.assert_array_equal(before, after)


def test_visible_landmarks_exclude_missing_outside_and_masked_joints():
    mask = np.zeros((30, 30), bool)
    mask[10:20, 10:20] = True
    valid = visible_landmarks([[15, 15], [np.nan, np.nan], [-1, 10], [0, 0]], mask)
    assert valid.tolist() == [True, False, False, False]


def test_window_reads_original_frame_paths_and_keeps_world_origin(
    tmp_path, monkeypatch
):
    from real2sim2real.egocentric import interaction_io

    monkeypatch.setattr(
        interaction_io, "load", lambda *a, **kw: {"metadata": {"units": "metres"}}
    )
    poses = np.tile(np.eye(4), (4, 1, 1))
    poses[:, 0, 3] = np.arange(4)
    (tmp_path / "timeline.json").write_text(
        json.dumps(
            dict(
                frames=4,
                frame_ids=[10, 11, 12, 13],
                timestamps_s=[0, 0.05, 0.1, 0.15],
                intrinsics=np.eye(3).tolist(),
            )
        )
    )
    np.savez(
        tmp_path / "geometry.npz",
        accepted=[False, True, True, True],
        world_from_camera=poses,
        world_from_object=poses,
        frame_ids=[10, 11, 12, 13],
        timestamps_s=[0, 0.05, 0.1, 0.15],
    )
    (tmp_path / "rgb").mkdir()
    Image.fromarray(np.full((4, 4, 3), 120, np.uint8)).save(tmp_path / "rgb/000002.png")
    source = Reconstruction(tmp_path, 2, start=2)
    assert source.frame_ids.tolist() == [12, 13]
    assert source.times.tolist() == [0.1, 0.15]
    assert source.world_from_camera[0, 0, 3] == 2
    assert (source.rgb(0) == 120).all()
    with pytest.raises(ValueError):
        Reconstruction(tmp_path, 3, start=2)


def test_archive_subset_does_not_slice_static_two_hand_fields(tmp_path):
    path = tmp_path / "source.npz"
    out = tmp_path / "selected.npz"
    faces = np.arange(18).reshape(2, 3, 3)
    np.savez(
        path,
        frame_ids=[40, 41],
        timestamps_s=[2, 2.05],
        faces=faces,
        accepted=np.ones((2, 2), bool),
        fixed_betas=np.ones((2, 10)),
    )
    subset_archive(path, out, [41], [2.05])
    with np.load(out) as data:
        np.testing.assert_array_equal(data["faces"], faces)
        assert data["accepted"].shape == (1, 2)
        assert data["fixed_betas"].shape == (2, 10)


def test_visual_anchors_are_camera_calibrated_and_missing_pose_is_not_scored():
    from real2sim2real.egocentric.hand_review_inputs import annotation_errors

    joints = np.zeros((1, 2, 21, 3))
    joints[..., 2] = 1
    view_from_target = np.eye(4)
    view_from_target[0, 3] = 0.1
    views = {
        "side": dict(
            intrinsics=[[100, 0, 50], [0, 100, 50], [0, 0, 1]],
            view_from_target=view_from_target,
            images={"40": {"sha256": "image-hash"}},
        )
    }
    label = dict(
        frame_id=40,
        view="side",
        hand=0,
        joint=4,
        xy_px=[60, 50],
        visibility="visible",
        image_sha256="image-hash",
    )
    assert (
        annotation_errors({"points": [label]}, joints, [40], views)[0]["error_px"] == 0
    )
    joints[0, 0] = np.nan
    assert (
        annotation_errors({"points": [label]}, joints, [40], views)[0]["error_px"]
        is None
    )
    assert (
        annotation_errors(
            {"points": [dict(label, visibility="occluded")]}, joints, [40], views
        )
        == []
    )
    with pytest.raises(ValueError, match="reviewed image"):
        annotation_errors(
            {"points": [dict(label, image_sha256="wrong")]}, joints, [40], views
        )


def test_model_runtime_excludes_visual_evaluation_annotations(tmp_path):
    from real2sim2real.egocentric.hand_experiment import runtime_config

    settings, phase = runtime_config(
        dict(
            output=str(tmp_path),
            reconstruction="/capture",
            frame_start=40,
            frame_count=46,
            annotations="/private/evaluation/annotations.json",
        ),
        "hawor",
    )
    assert phase == "hawor-hands"
    assert settings["read_only_inputs"] == ["/capture", str(tmp_path / "inputs")]
    assert "annotations" not in settings["model_options"]
    assert settings["output"] == str(tmp_path / "hawor")


def test_paired_temporal_scores_share_edges_and_exclude_rejected_spikes():
    from real2sim2real.egocentric.hand_experiment_review import paired_temporal

    a = np.zeros((7, 2, 21, 3))
    b = a.copy()
    a[3, 0] = 10
    accepted = np.ones((7, 2), bool)
    accepted[3, 0] = False
    result = paired_temporal(
        {
            "hamer": dict(world_joints=a, geometry_supported=accepted),
            "hawor": dict(world_joints=b, geometry_supported=np.ones_like(accepted)),
        },
        np.arange(7) * 0.05,
    )
    for name in ["hamer", "hawor"]:
        assert result[name][0]["shared_triples"] == 2
        assert result[name][0]["acceleration_rms_m_s2"] == 0
