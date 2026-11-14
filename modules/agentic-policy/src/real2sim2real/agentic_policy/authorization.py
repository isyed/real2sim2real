"""Explicit authorization for a frozen external-author campaign, not inference."""

import hashlib
import json


def scope(state):
    payload = {
        "bundle_hashes": state["bundle_hashes"],
        "author": state["config"].get("author", {}),
        "total_proposals": state["total_proposals"],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def require_authorization(root, state):
    if not state["config"].get("author", {}).get("network", False):
        return
    path = root / "author-authorization.json"
    if not path.is_file():
        raise PermissionError(
            "External author requires authorization for this frozen campaign"
        )
    receipt = json.loads(path.read_text())
    if (
        receipt.get("approved") is not True
        or receipt.get("scope_sha256") != scope(state)
        or not receipt.get("user_decision_reference")
    ):
        raise PermissionError("Missing or mismatched external-author authorization")


def review_requirement(state):
    """Identify the exact preceding outcome a human must review before revising."""
    rounds = state.get("rounds", [])
    if not rounds:
        return None
    previous = rounds[-1]
    if previous.get("status") != "evaluated":
        raise PermissionError("Complete the preceding evaluation before human review")
    result_hash = hashlib.sha256(
        json.dumps(previous["result"], sort_keys=True).encode()
    ).hexdigest()
    return {
        "next_proposal": f"P{len(rounds)}",
        "previous_proposal": previous["id"],
        "previous_result_sha256": result_hash,
        "scope_sha256": scope(state),
    }


def require_revision_review(root, state):
    """Optional hard gate: overall campaign authorization cannot approve later reviews."""
    if (
        not state["config"]
        .get("author", {})
        .get("human_review_between_proposals", False)
    ):
        return
    expected = review_requirement(state)
    if expected is None:
        return  # Initial author call is covered by the explicit campaign launch.
    path = root / "human-reviews" / (expected["next_proposal"] + ".json")
    if not path.is_file():
        raise PermissionError(
            "Human review of the previous result is required before another author call"
        )
    receipt = json.loads(path.read_text())
    if (
        receipt.get("approved") is not True
        or not receipt.get("user_decision_reference")
        or any(receipt.get(k) != v for k, v in expected.items())
    ):
        raise PermissionError("Missing or stale human review for this exact result")


def approved_revision_feedback(root, state):
    """Return the verified human decision for the next author, or None initially."""
    if (
        not state["config"]
        .get("author", {})
        .get("human_review_between_proposals", False)
    ):
        return None
    require_revision_review(root, state)
    expected = review_requirement(state)
    if expected is None:
        return None
    return json.loads(
        (root / "human-reviews" / (expected["next_proposal"] + ".json")).read_text()
    )
