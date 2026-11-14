import copy
import json

import pytest
from real2sim2real.agentic_policy import campaign
from real2sim2real.agentic_policy.authorization import require_authorization, scope
from real2sim2real.agentic_policy.extensions import extend_budget
from real2sim2real.agentic_policy.storage import hashes, load

from real2sim2real.artifacts import write_json


def prepared(tmp_path, monkeypatch, complete=True):
    bundle = tmp_path / "bundle"
    (bundle / "author").mkdir(parents=True)
    write_json(
        bundle / "experiment.json",
        {
            "mode": "manual",
            "feedback_revisions": 0,
            "instruction": "test",
            "rollout_timeout_s": 10,
            "backend": {"kind": "diagnostic_counter"},
            "author": {"network": True},
        },
    )
    root = tmp_path / "campaign"
    campaign.initialize(bundle, root)
    policy = tmp_path / "policy"
    policy.mkdir()
    write_json(policy / "workflow.json", {})
    monkeypatch.setattr(
        campaign,
        "rollout",
        lambda *a: {"infrastructure_error": None, "verdict": {"task_completed": True}},
    )
    if complete:
        campaign.submit(root, policy)
    return root, policy


def test_extension_preserves_evidence_and_invalidates_old_authorization(
    tmp_path, monkeypatch
):
    root, policy = prepared(tmp_path, monkeypatch)
    before = copy.deepcopy(load(root))
    frozen = hashes(root / "bundle")
    write_json(
        root / "author-authorization.json",
        {
            "approved": True,
            "scope_sha256": scope(before),
            "user_decision_reference": "initial approval",
        },
    )
    require_authorization(root, before)
    amendment = extend_budget(root, 4, "User requested four more revisions")
    after = load(root)
    assert after["total_proposals"] == 5
    assert after["config"] == before["config"]
    assert after["rounds"] == before["rounds"]
    assert after["retained_nominal"] == "P0"
    assert hashes(root / "bundle") == frozen
    assert amendment["previous_scope_sha256"] != amendment["scope_sha256"]
    with pytest.raises(PermissionError, match="mismatched"):
        require_authorization(root, after)
    for _ in range(4):
        campaign.submit(root, policy)
    assert [r["id"] for r in load(root)["rounds"]] == [f"P{i}" for i in range(5)]
    with pytest.raises(ValueError, match="budget"):
        campaign.submit(root, policy)


def test_cannot_extend_unfinished_campaign_or_active_writer(tmp_path, monkeypatch):
    from real2sim2real.agentic_policy.storage import locked

    root, _ = prepared(tmp_path, monkeypatch, complete=False)
    before = (root / "state.json").read_bytes()
    with pytest.raises(ValueError, match="Finish"):
        extend_budget(root, 4, "requested")
    with locked(root), pytest.raises(BlockingIOError):
        extend_budget(root, 4, "requested")
    assert (root / "state.json").read_bytes() == before


@pytest.mark.parametrize(
    "count,reason", [(0, "x"), (-1, "x"), (True, "x"), (1, ""), (101, "x")]
)
def test_invalid_extension_does_not_change_campaign(
    tmp_path, monkeypatch, count, reason
):
    root, _ = prepared(tmp_path, monkeypatch)
    before = json.loads((root / "state.json").read_text())
    with pytest.raises(ValueError):
        extend_budget(root, count, reason)
    assert load(root) == before
