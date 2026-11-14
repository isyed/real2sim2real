import json

import numpy as np
import pytest
from real2sim2real.agentic_policy import campaign
from real2sim2real.agentic_policy.recovery import record_clock_rejection
from real2sim2real.agentic_policy.storage import hashes, load

from real2sim2real.artifacts import write_json


def failed_audit(
    tmp_path, monkeypatch, times, fault="simulation_clock_reset_or_invalid_step"
):
    bundle = tmp_path / "bundle"
    (bundle / "author").mkdir(parents=True)
    write_json(
        bundle / "experiment.json",
        {
            "mode": "manual",
            "feedback_revisions": 1,
            "instruction": "test",
            "rollout_timeout_s": 10,
            "backend": {"kind": "diagnostic_counter"},
        },
    )
    root = tmp_path / "campaign"
    campaign.initialize(bundle, root)
    policy = tmp_path / "policy"
    policy.mkdir()
    write_json(policy / "workflow.json", {})

    def interrupted(bundle, config, candidate, output):
        (output / "physics").mkdir()
        (output / "work").mkdir()
        write_json(output / "physics/status.json", {"fault": fault})
        write_json(output / "work/graph-result.json", {"graph_status": "failure"})
        write_json(output / "physics/observations.json", [{"time_s": t} for t in times])
        np.savez(output / "physics/states.npz", timestamps_s=times)
        raise ValueError(
            "Finite matching sampled states with a strictly increasing clock required"
        )

    monkeypatch.setattr(campaign, "rollout", interrupted)
    campaign.submit(root, policy)
    return root


def test_reported_clock_failure_is_rejection_without_replaying_or_modifying_evidence(
    tmp_path, monkeypatch
):
    root = failed_audit(tmp_path, monkeypatch, [0.0, 0.02, 0.04, 0.001])
    trial = root / "rounds/P0/trial-0"
    before = hashes(trial / "physics")
    error = (trial / "failure.json").read_bytes()
    result = record_clock_rejection(root)
    assert not result["verdict"]["task_completed"]
    assert result["verdict"]["diagnostics"] == {}
    assert not result["assessment"]["nominal_completed"]
    assert result["numerical_rejection"]["last_sample_before_reset_s"] == 0.04
    assert load(root)["rounds"][0]["status"] == "evaluated"
    assert hashes(trial / "physics") == before
    assert (trial / "failure.json").read_bytes() == error
    with pytest.raises(ValueError, match="interrupted"):
        record_clock_rejection(root)


@pytest.mark.parametrize(
    "times,fault",
    [
        ([0.0, 0.02, 0.04], "simulation_clock_reset_or_invalid_step"),
        ([0.0, 0.02, 0.001], "unrelated"),
    ],
)
def test_unrelated_infrastructure_errors_are_not_reclassified(
    tmp_path, monkeypatch, times, fault
):
    root = failed_audit(tmp_path, monkeypatch, times, fault)
    before = json.loads((root / "state.json").read_text())
    with pytest.raises(ValueError):
        record_clock_rejection(root)
    assert load(root) == before
