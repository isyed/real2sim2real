import subprocess

import numpy as np
import pytest

from real2sim2real.egocentric.hand_audit import intervals
from real2sim2real.egocentric.hand_refinement import (
    admit_temporal,
    refinement_rejections,
)
from real2sim2real.egocentric.metric_hands import hand_rejections
from real2sim2real.egocentric.mano_refinement import validate_options
from real2sim2real.egocentric.runtime import run_model


LIMITS = dict(
    max_reprojection_median_px=12,
    max_reprojection_p90_px=25,
    max_validation_surface_median_m=0.012,
    max_validation_surface_p90_m=0.03,
    min_validation_points=80,
)


def measurements(**changes):
    return dict(
        dict(
            reprojection_median_px=10.0,
            reprojection_p90_px=24.0,
            validation_surface_median_m=0.005,
            validation_surface_p90_m=0.015,
            validation_points=100,
        ),
        **changes,
    )


def test_refinement_cannot_relax_parent_limits_or_trade_depth_for_image_fit():
    before = measurements(reprojection_p90_px=35.0)
    good = measurements(reprojection_p90_px=20.0)
    assert not refinement_rejections(before, good, LIMITS)
    depth_regression = dict(good, validation_surface_median_m=0.009)
    assert "depth_regression:validation_surface_median_m" in refinement_rejections(
        before, depth_regression, LIMITS
    )
    image_failure = dict(good, reprojection_p90_px=26.0)
    assert "reprojection_p90_px" in refinement_rejections(before, image_failure, LIMITS)
    assert "insufficient_image_improvement" in refinement_rejections(good, good, LIMITS)


def test_nonfinite_or_missing_depth_cannot_pass_shared_hand_gates():
    assert "reprojection_median_px" in hand_rejections(
        measurements(reprojection_median_px=np.nan), LIMITS
    )
    assert "validation_surface_p90_m" in hand_rejections(
        measurements(validation_surface_p90_m=None), LIMITS
    )
    assert "validation_coverage" in hand_rejections(
        measurements(validation_points=79), LIMITS
    )
    with pytest.raises(ValueError, match="finite parent"):
        refinement_rejections(
            measurements(validation_surface_median_m=None), measurements(), LIMITS
        )


@pytest.mark.parametrize(
    "options",
    [
        {"unknown": 1},
        {"max_joint_correction_deg": 46},
        {"max_rigid_correction_deg": float("nan")},
        {"max_translation_correction_m": 0.051},
        {"max_translation_correction_m": True},
        {"correspondence_iterations": 0},
        {"correspondence_iterations": 5.0},
        {"correspondence_iterations": True},
        {"correspondence_iterations": 21},
    ],
)
def test_optimizer_cannot_request_unbounded_or_misspelled_options(options):
    with pytest.raises(ValueError):
        validate_options(options)


def test_optimizer_defaults_keep_corrections_local():
    options = validate_options({"max_joint_correction_deg": 20})
    assert options == dict(
        max_joint_correction_deg=20,
        max_rigid_correction_deg=15,
        max_translation_correction_m=0.03,
        correspondence_iterations=5,
    )


def test_temporal_admission_preserves_anchors_and_rejects_disruptive_insertions():
    parent = np.zeros((5, 2), bool)
    parent[[0, 4], 0] = True
    proposed = np.zeros_like(parent)
    proposed[1:4, 0] = True
    wrist = np.zeros((5, 2, 3))
    wrist[:, 0, 0] = [0.0, 0.01, 0.5, 0.03, 0.04]
    accepted = admit_temporal(parent, proposed, wrist, np.arange(5) * 0.1, 1.0)
    assert accepted[parent].all()
    assert not accepted[2, 0]
    assert not accepted[:, 1].any()


def test_temporal_admission_never_invents_missing_hands():
    parent = np.zeros((3, 2), bool)
    parent[0, 0] = True
    wrists = np.full((3, 2, 3), np.nan)
    wrists[0, 0] = 0.0
    assert np.array_equal(
        admit_temporal(parent, np.zeros_like(parent), wrists, [0, 0.1, 0.2], 3), parent
    )
    with pytest.raises(ValueError, match="timeline"):
        admit_temporal(parent, parent, wrists, [0, 0, 0.2], 3)


def test_audit_uses_original_frame_ids_and_does_not_join_separate_gaps():
    rows = intervals([True, True, False, True], [40, 41, 42, 43], [0.0, 0.1, 0.2, 0.3])
    assert [(r["first_frame"], r["last_frame"], r["frames"]) for r in rows] == [
        (40, 41, 2),
        (43, 43, 1),
    ]
    assert rows[1]["start_s"] == 0.3
    assert intervals([False], [7], [0.0]) == []


@pytest.mark.parametrize("timeout", [False, True])
def test_cpu_refinement_has_no_gpu_and_cleans_its_container(
    tmp_path, monkeypatch, timeout
):
    root = tmp_path / "source"
    root.mkdir()
    out = tmp_path / "out"
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if command[:2] == ["docker", "run"] and timeout:
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", run)
    config = dict(
        source_root=str(root),
        weights={"hamer": str(tmp_path / "model")},
        images={"refine-hands": "local-test-image"},
        timeouts={"refine-hands": 60},
    )
    if timeout:
        with pytest.raises(subprocess.TimeoutExpired):
            run_model(config, out, "refine-hands")
    else:
        run_model(config, out, "refine-hands")
    command = calls[0]
    assert "--gpus" not in command and "CUDA_VISIBLE_DEVICES=" in command
    assert "NVIDIA_VISIBLE_DEVICES=void" in command and "runc" in command
    assert "--check" not in command
    name = command[command.index("--name") + 1]
    assert calls[-1] == ["docker", "rm", "-f", name]
