import numpy as np
import pytest

from real2sim2real.egocentric.hand_sequence import (
    sequence_options,
    select_components,
    derivatives,
    compare_motion,
    motion_rejections,
    protected_boundary_edges,
)
from real2sim2real.egocentric.point_hand_experiment import runtime_config
from real2sim2real.egocentric.point_hand_review import diagnostic_support


def test_rejected_context_pose_is_not_colored_as_accepted_baseline():
    baseline = dict(accepted=np.ones((2, 2), bool))
    candidate = dict(
        sequence_selected=np.array([[True, False], [False, False]]),
        refinement_attempted=np.zeros((2, 2), bool),
        diagnostic_geometry_supported=np.zeros((2, 2), bool),
    )
    assert diagnostic_support(candidate, baseline).tolist() == [
        [False, True],
        [True, True],
    ]


def test_derivatives_use_nonuniform_capture_times():
    times = np.array([0, 0.04, 0.11, 0.16])
    joints = np.zeros((4, 21, 3))
    joints[:, :, 0] = times[:, None] ** 2
    _, acceleration = derivatives(joints, times)
    np.testing.assert_allclose(acceleration[..., 0], 2)
    np.testing.assert_array_equal(acceleration[..., 1:], 0)


def test_boundary_velocity_prior_never_preserves_recovery_jump():
    selected = np.array([False, True, True, True, False])
    baseline = np.array([True, True, False, True, True])
    assert protected_boundary_edges(selected, baseline).tolist() == [
        True,
        False,
        False,
        True,
    ]
    assert not protected_boundary_edges(selected, ~selected).any()


def test_components_keep_measured_anchors_and_merge_shared_triplets():
    observed = np.ones((12, 2), bool)
    recovered = np.zeros_like(observed)
    recovered[[3, 4, 6], 0] = True
    selected, groups = select_components(recovered, observed, np.arange(12) * 0.05)
    assert len(groups) == 1
    assert groups[0]["selected"].tolist() == [3, 4, 6]
    assert groups[0]["window"].tolist() == list(range(1, 9))
    np.testing.assert_array_equal(selected, recovered)


def test_context_and_anchor_windows_do_not_bridge_missing_observations_or_time():
    observed = np.ones((12, 2), bool)
    observed[5, 0] = False
    recovered = np.zeros_like(observed)
    recovered[3, 0] = recovered[8, 0] = True
    times = np.arange(12) * 0.05
    times[9:] += 1
    selected, groups = select_components(recovered, observed, times, context_frames=3)
    assert not selected[5, 0] and not selected[9:, 0].any()
    assert len(groups) == 2
    assert all(5 not in g["window"] and 9 not in g["window"] for g in groups)


def test_stationary_wrist_does_not_hide_a_finger_jump():
    before = np.zeros((5, 21, 3))
    before[2, 4, 0] = 0.1
    after = before.copy()
    after[2, 4, 0] = 0.03
    selected = np.array([False, False, True, False, False])
    comparison = compare_motion(before, after, selected, np.arange(5) * 0.05)
    assert (
        comparison["after"]["acceleration_max_m_s2"]
        < comparison["before"]["acceleration_max_m_s2"]
    )
    assert not motion_rejections(comparison)
    reverse = compare_motion(after, before, selected, np.arange(5) * 0.05)
    assert "sequence_boundary_regression" in motion_rejections(reverse)


@pytest.mark.parametrize(
    "options",
    [
        dict(context_frames=True),
        dict(temporal_weight=0),
        dict(max_joint_delta_deg=46),
        dict(max_translation_delta_m=0.5),
        dict(unknown=2),
    ],
)
def test_invalid_sequence_options_are_rejected(options):
    with pytest.raises(ValueError):
        sequence_options(options)


def test_sequence_runtime_isolated_from_evaluation_and_preserves_parent(tmp_path):
    config = dict(
        output=str(tmp_path / "run"),
        parent=str(tmp_path / "parent"),
        reconstruction=str(tmp_path / "reconstruction"),
        frame_start=40,
        frame_count=46,
        heldout_frame_ids=[50],
        mask_ids=[2, 3],
        object_id=1,
        weights=dict(tapir="/weights/tapir", hamer="/weights/hamer"),
        candidate_directory="sequence-test",
    )
    settings, phase = runtime_config(config, "sequence")
    assert phase == "sequence-point-hands"
    assert settings["output"].endswith("/sequence-test")
    assert all("evaluation" not in p for p in settings["read_only_inputs"])
    assert config["output"] not in settings["read_only_inputs"]
    config["candidate_directory"] = "refined"
    with pytest.raises(ValueError):
        runtime_config(config, "sequence")
