from copy import deepcopy

import numpy as np
import pytest
from real2sim2real.controller_actor_training.guided_motion import select_interval
from real2sim2real.controller_actor_training.reference_extension import (
    effective_parent,
    reference_transforms,
)

from real2sim2real.artifacts import digest


def archive(n=5):
    poses = np.tile(np.eye(4), (n, 1, 1))
    poses[:, 0, 3] = np.arange(n) * 0.01
    return dict(
        timestamps_s=3 + np.arange(n) * 0.05,
        world_from_object=poses,
        frame_ids=np.arange(n) + 40,
    )


def test_requested_interval_cannot_silently_truncate():
    ref = select_interval(archive(), 40, 44)
    assert ref.frame_ids.tolist() == [40, 41, 42, 43, 44]
    for start, end in [(39, 43), (40, 45), (40, 40), (True, 44)]:
        with pytest.raises(ValueError):
            select_interval(archive(), start, end)
    a = archive()
    a["frame_ids"][2] = 41
    with pytest.raises(ValueError, match="incomplete"):
        select_interval(a, 40, 44)


def save(tmp_path, name, values):
    path = tmp_path / name
    np.savez(path, **values)
    return dict(
        path=str(path),
        sha256=digest(path),
        mode="world_increment_after_grasp",
        limits=dict(
            maximum_translation_m=1,
            maximum_rotation_rad=1,
            maximum_speed_m_s=1,
            maximum_angular_speed_rad_s=1,
        ),
    )


def test_extension_preserves_prefix_clock_relative_grasp_and_input(tmp_path):
    old = save(tmp_path, "old.npz", archive(3))
    new = save(tmp_path, "new.npz", archive())
    anchor = np.eye(4)
    anchor[2, 3] = 0.2
    parent = dict(
        configuration=dict(reference_guidance=old),
        reference_anchor_world_from_object=anchor.tolist(),
        records=[
            dict(phase="approach", time_s=1),
            dict(
                phase="hold",
                time_s=2,
                reference_elapsed_s=0.05,
                reference_position_m=[0.01, 0, 0.2],
                reference_rotation=np.eye(3).tolist(),
                object_position_m=[0.013, 0, 0.2],
                object_rotation=np.eye(3).tolist(),
            ),
            dict(
                phase="hold",
                time_s=3.3,
                reference_elapsed_s=1.3,
                reference_position_m=[0.02, 0, 0.2],
                reference_rotation=np.eye(3).tolist(),
                object_position_m=[0.023, 0, 0.2],
                object_rotation=np.eye(3).tolist(),
            ),
        ],
    )
    before = deepcopy(parent)
    changes, overrides = reference_transforms(parent, new)
    np.testing.assert_allclose(changes[0], np.eye(4))
    np.testing.assert_allclose(changes[1], np.eye(4), atol=1e-12)
    assert changes[2, 0, 3] == pytest.approx(0.02)
    assert parent == before
    audit = dict(
        config=dict(reference_extension=new), extended_reference_records=overrides
    )
    effective = effective_parent(parent, audit)
    assert effective["records"][2]["reference_position_m"][0] == pytest.approx(0.04)
    assert effective["records"][2]["object_position_m"][0] == pytest.approx(0.043)
    assert effective["records"][2]["time_s"] == 3.3 and parent == before
    overrides[2]["time_s"] = 9
    with pytest.raises(ValueError, match="phase, clock or fingers"):
        effective_parent(parent, audit)
    modified = archive()
    modified["world_from_object"][0, 0, 3] = 0.001
    bad = save(tmp_path, "bad.npz", modified)
    with pytest.raises(ValueError, match="preserve"):
        reference_transforms(parent, bad)
    short = deepcopy(parent)
    short["records"][-1]["reference_elapsed_s"] = 0.3
    with pytest.raises(ValueError, match="clock"):
        reference_transforms(short, new)


def test_longer_hold_is_explicit_and_does_not_retime_reference():
    from real2sim2real.controller_actor_training.reference_extension import (
        declared_hold,
    )

    assert declared_hold(10, 10.27, 12) == 12
    assert declared_hold(10, 2.2) == 10
    for hold in (None, 9, 10, True, float("nan"), 26):
        with pytest.raises(ValueError):
            declared_hold(10, 10.27, hold)


def test_prepending_capture_frames_is_a_declared_new_origin_not_a_prefix_claim():
    from real2sim2real.controller_actor_training.guided_motion import MotionReference
    from real2sim2real.controller_actor_training.reference_extension import match_window

    old = archive(3)
    new = archive(7)
    new["frame_ids"] -= 2
    new["timestamps_s"] -= 0.1
    new["world_from_object"][:, 0, 3] -= 0.02

    def ref(a):
        return MotionReference(
            a["timestamps_s"], a["world_from_object"], a["frame_ids"]
        )

    with pytest.raises(ValueError):
        match_window(ref(old), ref(new))
    result = match_window(ref(old), ref(new), allow_prepend=True)
    assert (
        result["prefix_preserved"] is False
        and result["reference_origin_changed"] is True
    )
    assert result["capture_origin_shift_s"] == pytest.approx(0.1)
    assert result["original_capture_start_pose_reproduced"] is False
    new["world_from_object"][3, 0, 3] += 0.001
    with pytest.raises(ValueError):
        match_window(ref(old), ref(new), allow_prepend=True)
