import json
from copy import deepcopy

import pytest
from real2sim2real.agentic_policy.authorization import (
    require_revision_review,
    review_requirement,
)


def fixture():
    return {
        "bundle_hashes": {"scene": "fixed"},
        "config": {"author": {"human_review_between_proposals": True}},
        "total_proposals": 3,
        "rounds": [{"id": "P0", "status": "evaluated", "result": {"nominal": False}}],
    }


def test_review_is_required_for_exact_result_before_next_author(tmp_path):
    state = fixture()
    with pytest.raises(PermissionError):
        require_revision_review(tmp_path, state)
    (tmp_path / "human-reviews").mkdir()
    receipt = review_requirement(state) | {
        "approved": True,
        "user_decision_reference": "Explicit test user decision",
    }
    (tmp_path / "human-reviews/P1.json").write_text(json.dumps(receipt))
    require_revision_review(tmp_path, state)
    changed = deepcopy(state)
    changed["rounds"][0]["result"]["nominal"] = True
    with pytest.raises(PermissionError):
        require_revision_review(tmp_path, changed)
    state["rounds"].append(
        {"id": "P1", "status": "evaluated", "result": {"nominal": False}}
    )
    with pytest.raises(PermissionError):
        require_revision_review(tmp_path, state)


def test_initial_call_and_unfinished_result(tmp_path):
    state = fixture()
    state["rounds"] = []
    require_revision_review(tmp_path, state)
    state = fixture()
    state["rounds"][0]["status"] = "running"
    with pytest.raises(PermissionError):
        require_revision_review(tmp_path, state)


def test_verified_human_feedback_available_to_author(tmp_path):
    from real2sim2real.agentic_policy.authorization import approved_revision_feedback

    state = fixture()
    with pytest.raises(PermissionError):
        approved_revision_feedback(tmp_path, state)
    (tmp_path / "human-reviews").mkdir()
    receipt = review_requirement(state) | {
        "approved": True,
        "user_decision_reference": "Explicit outcome approval",
        "feedback": "Improve retention with unchanged trusted limits",
    }
    (tmp_path / "human-reviews/P1.json").write_text(json.dumps(receipt))
    assert approved_revision_feedback(tmp_path, state) == receipt
    state["rounds"] = []
    assert approved_revision_feedback(tmp_path, state) is None
