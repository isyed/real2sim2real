import numpy as np
import pytest
from real2sim2real.controller_actor_training.arm_reference_evaluation import score_poses


def test_object_metrics_include_rotation_and_are_not_finger_metrics():
    reference = np.array([[0.0, 0, 0, 0, 0, 0, 1]])
    actual = reference.copy()
    actual[0, 0] = 0.02
    vertices = np.eye(3) * 0.05
    score = score_poses(actual, reference, vertices)
    assert score["mppe_cm"] == pytest.approx(2)
    assert score["peak_translation_error_m"] == pytest.approx(0.02)
    actual = np.array([[0.0, 0, 0, 0, 0, 1, 0]])
    rotated = score_poses(actual, reference, vertices)
    assert rotated["peak_translation_error_m"] == 0
    assert rotated["mppe_cm"] > 0
    assert rotated["add_auc"] < 1


def test_empty_cohort_cannot_look_perfect():
    with pytest.raises(ValueError):
        score_poses(np.empty((0, 7)), np.empty((0, 7)), np.eye(3))


def test_stationary_diagnostic_cannot_receive_demonstrated_motion_scores(tmp_path):
    import json
    from real2sim2real.controller_actor_training.arm_reference_evaluation import (
        evaluate,
    )

    (tmp_path / "result.json").write_text(
        json.dumps(dict(motion_condition="stationary_hold"))
    )
    with pytest.raises(ValueError, match="Stationary"):
        evaluate(tmp_path)
