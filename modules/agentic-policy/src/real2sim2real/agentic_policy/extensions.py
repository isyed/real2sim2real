"""Explicit budget amendments without modifying frozen experimental inputs."""

import time
from pathlib import Path

from .authorization import scope
from .storage import load, locked, store


def extend_budget(root, additional_revisions, reason):
    """Append slots after a finished campaign; external execution needs a new scope.

    The caller must have authorization for the extension. This operation neither
    invokes an author nor writes an approval receipt. Final held-out results must
    remain outside feedback, including when extending an earlier search budget.
    """
    if type(additional_revisions) is not int or additional_revisions < 1:
        raise ValueError("additional_revisions must be a positive integer")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("An explicit amendment reason is required")
    root = Path(root).resolve()
    with locked(root):
        state = load(root)
        previous = state["total_proposals"]
        if (
            state["pending"]
            or len(state["rounds"]) != previous
            or any(r["status"] != "evaluated" for r in state["rounds"])
            or any(a["status"] == "running" for a in state["author_attempts"])
        ):
            raise ValueError("Finish the existing proposal budget before extending")
        total = previous + additional_revisions
        if total > 101:
            raise ValueError("At most 101 total proposals are supported")
        old_scope = scope(state)
        state["total_proposals"] = total
        amendment = {
            "previous_total": previous,
            "total_proposals": total,
            "additional_revisions": additional_revisions,
            "reason": reason.strip(),
            "timestamp_unix_s": time.time(),
            "previous_scope_sha256": old_scope,
            "scope_sha256": scope(state),
        }
        state.setdefault("budget_amendments", []).append(amendment)
        store(root, state)
        return amendment
