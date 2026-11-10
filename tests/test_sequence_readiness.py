import numpy as np
import pytest

from real2sim2real.egocentric.interaction_readiness import sequence_assessment


def inputs():
    data = dict(
        frame_ids=np.arange(3),
        geometry_supported=np.ones((3, 2), bool),
        sequence_selected=np.ones((3, 2), bool),
        sequence_admitted=np.ones((3, 2), bool),
    )
    quality = dict(
        sequence_fitting=True,
        sequence_selected_per_hand=[3, 3],
        sequence_admitted_per_hand=[3, 3],
        components=[dict(hand=0, frame_ids=[0, 1, 2], accepted=True)],
    )
    evidence = {
        key: dict(passed=True, source="synthetic", reason="Synthetic passing evidence")
        for key in [
            "hand_identity",
            "independent_alignment",
            "depth_support",
            "contact",
        ]
    }
    return data, quality, evidence


def test_complete_fallback_coverage_does_not_hide_failed_sequence():
    data, quality, evidence = inputs()
    data["sequence_admitted"][:, 0] = False
    quality["sequence_admitted_per_hand"] = [0, 3]
    quality["components"][0].update(
        accepted=False,
        geometry_failures=[dict(frame_id=1, reasons=["reprojection_p90_px"])],
    )
    result = sequence_assessment(
        data, quality, required_frame_ids=[0, 1, 2], evidence=evidence
    )
    assert result["retained_geometry_frames_per_hand"] == [3, 3]
    assert result["admitted_frames_per_hand"] == [0, 3]
    assert result["decision"] == "not_ready_for_imitation_control"
    assert "rejected_sequence_components" in result["blockers"]


def test_passing_reference_never_claims_controller_validation():
    data, quality, evidence = inputs()
    result = sequence_assessment(
        data, quality, required_frame_ids=[0, 1, 2], evidence=evidence
    )
    assert result["reference_ready_for_retargeting"]
    assert not result["control_ready"]
    assert result["remaining_robot_checks"]


@pytest.mark.parametrize("status", [False, None])
def test_uncertain_evidence_blocks_reference_promotion(status):
    data, quality, evidence = inputs()
    evidence["hand_identity"]["passed"] = status
    result = sequence_assessment(
        data, quality, required_frame_ids=[0, 1, 2], evidence=evidence
    )
    assert "unvalidated_evidence:hand_identity" in result["blockers"]
    assert not result["reference_ready_for_retargeting"]


def test_missing_evidence_and_unassessed_frames_are_explicit():
    data, quality, evidence = inputs()
    del evidence["contact"]
    result = sequence_assessment(
        data, quality, required_frame_ids=[0, 1, 2, 3], evidence=evidence
    )
    assert result["missing_frame_ids"] == [3]
    assert "missing_evidence:contact" in result["blockers"]
    assert "requested_frames_not_assessed" in result["blockers"]


def test_inconsistent_artifact_counts_are_rejected():
    data, quality, evidence = inputs()
    quality["sequence_admitted_per_hand"] = [0, 3]
    with pytest.raises(ValueError, match="counts differ"):
        sequence_assessment(
            data, quality, required_frame_ids=[0, 1, 2], evidence=evidence
        )
