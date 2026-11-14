import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from real2sim2real.agentic_policy import campaign, execution
from real2sim2real.agentic_policy.proposal_history import event_trace, validate_options
from real2sim2real.agentic_policy.reference_refinement import ReferenceRefinementBackend
from real2sim2real.agentic_policy.storage import load

from real2sim2real.artifacts import write_json


def setup(tmp_path, options=None):
    bundle = tmp_path / "bundle"
    (bundle / "author").mkdir(parents=True)
    (bundle / "seed").mkdir()
    config = {
        "mode": "manual",
        "feedback_revisions": 5,
        "instruction": "test",
        "rollout_timeout_s": 5,
        "backend": {"kind": "diagnostic_counter"},
        "seed": "seed",
    }
    if options:
        config["author_loop"] = options
    write_json(bundle / "experiment.json", config)
    root = tmp_path / "campaign"
    campaign.initialize(bundle, root)
    return root


def test_retain_decision_stops_without_physics_or_more_author_calls(
    tmp_path, monkeypatch
):
    root = setup(tmp_path)
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    write_json(
        candidate / "decision.json",
        {"action": "retain_baseline", "reason": "No supported hypothesis"},
    )
    monkeypatch.setattr(
        campaign, "rollout", lambda *a, **kw: pytest.fail("physics ran")
    )
    row = campaign.submit(root, candidate)
    assert row["status"] == "retained_baseline" and row["trials"] == []
    assert campaign.advance(root)["next"] == "stopped"
    with pytest.raises(ValueError, match="stopped"):
        campaign.submit(root, candidate)
    assert load(root)["retained_nominal"] is None


def test_ambiguous_decision_rejected_before_archive(tmp_path):
    root = setup(tmp_path)
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    write_json(
        candidate / "decision.json", {"action": "retain_baseline", "reason": "test"}
    )
    write_json(candidate / "workflow.json", {})
    with pytest.raises(ValueError, match="not both"):
        campaign.submit(root, candidate)
    assert not load(root)["rounds"]


def test_all_measured_history_survives_later_proposals(tmp_path, monkeypatch):
    root = setup(tmp_path, {"history": True})
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    write_json(candidate / "workflow.json", {})
    count = []

    def rollout(bundle, config, candidate, output):
        count.append(1)
        write_json(
            output / "diagnostics/observations.json",
            [{"time_s": 0, "pad_forces_n": [0], "penetration_m": 0}],
        )
        return {
            "infrastructure_error": None,
            "verdict": {"task_completed": len(count) == 1},
        }

    monkeypatch.setattr(campaign, "rollout", rollout)
    for _ in range(3):
        campaign.submit(root, candidate)
    (root / "private-heldout.json").write_text("excluded")
    work = tmp_path / "packet"
    campaign.preview_author(root, work)
    index = json.loads((work / "campaign-history/index.json").read_text())
    assert [r["id"] for r in index] == ["P0", "P1", "P2"]
    assert json.loads((work / "campaign-history/P0/result.json").read_text())[
        "verdict"
    ]["task_completed"]
    assert not json.loads((work / "campaign-history/P1/result.json").read_text())[
        "verdict"
    ]["task_completed"]
    assert not (work / "private-heldout.json").exists()
    assert (
        load(root)["retained_nominal"] == "P0"
    )  # A failed experiment does not replace it.


def test_event_windows_preserve_brief_contact_loss_and_bound_chatter():
    rows = [
        {"time_s": i * 0.02, "pad_forces_n": [0 if i == 7 else 1], "penetration_m": 0}
        for i in range(80)
    ]
    times = [x["time_s"] for x in event_trace(rows)["rows"]]
    assert all(i * 0.02 in times for i in range(5, 11))
    rows = [{"pad_forces_n": [i % 2], "penetration_m": 0} for i in range(5000)]
    trace = event_trace(rows)
    assert len(trace["rows"]) <= 300 and trace["omitted_event_samples"] > 0


def fingerprint_backend():
    backend = ReferenceRefinementBackend.__new__(ReferenceRefinementBackend)
    backend.bounds = {
        "maximum_knots": 12,
        "minimum_gain_scale": 0.5,
        "maximum_gain_scale": 2.0,
        "maximum_finger_offset_rad": 0.05,
        "maximum_offset_speed_rad_s": 0.2,
    }
    times = np.arange(0, 4, 0.005)
    backend.session = SimpleNamespace(
        config={
            "finger_joints": ["thumb"],
            "pressure_targets_n": [2.5],
            "pressure": {"gain_m_per_n_s": 0.002},
            "schedule": {"pressure_start_reference_s": 1.0, "release_reference_s": 3.0},
            "pressure_groups": [[0]],
            "feedback_s": 0.005,
        },
        ref={
            "reference_times_s": times,
            "finger_targets_rad": np.zeros((len(times), 1)),
        },
        model=SimpleNamespace(jnt_range=np.array([[-1, 1]])),
        hand={"joints": [0]},
    )
    return backend


def test_effective_fingerprint_ignores_redundant_knots_but_detects_posture():
    b = fingerprint_backend()
    assert b.edit_fingerprint({}) == b.edit_fingerprint(
        {"reference_times_s": [1.0, 2.0, 3.0]}
    )
    assert b.edit_fingerprint({}) != b.edit_fingerprint(
        {"reference_times_s": [1.0, 2.0, 3.0], "finger_offsets_rad": [[0], [0.02], [0]]}
    )
    with pytest.raises(ValueError):
        b.edit_fingerprint(
            {"reference_times_s": [1.0, 2.0, 3.0], "gain_scale": [1, 3, 1]}
        )


def test_actual_rpc_duplicate_stops_even_if_worker_would_retry(tmp_path, monkeypatch):
    backend = fingerprint_backend()
    backend.tools = {
        "controller.run_reference": lambda **kw: pytest.fail("physics must not run")
    }
    backend.finish = lambda *a: pytest.fail("no simulation diagnostics for duplicate")
    monkeypatch.setattr(execution, "make_backend", lambda *a: backend)
    monkeypatch.setattr(execution, "schemas", lambda *a: {})
    # Real pipe/protocol, harmless local subprocess replacing the isolated graph worker.
    script = 'import json,time; print(json.dumps({"tool":"controller.run_reference","inputs":{"edits":{}}}),flush=True);time.sleep(2)'
    monkeypatch.setattr(
        execution.sandbox, "command", lambda *a, **kw: [sys.executable, "-c", script]
    )
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    output = tmp_path / "out"
    output.mkdir()
    receipt = execution.rollout(
        tmp_path,
        {"backend": {}, "rollout_timeout_s": 5},
        candidate,
        output,
        duplicate_edit_fingerprints={},
        baseline_edits={},
    )
    assert receipt["duplicate_edit"]["matches_proposal"] == "declared_baseline"
    assert receipt["physics_executed"] is False and not (output / "physics").exists()


def test_duplicate_terminal_preserves_prior_result_and_stops_cases(
    tmp_path, monkeypatch
):
    root = setup(tmp_path)
    state = load(root)
    # Test the evaluated orchestration on a frozen config with no backend instantiation.
    from real2sim2real.agentic_policy.storage import hashes, store

    state["config"]["backend"] = {"kind": "reference_refinement"}
    state["config"]["author_loop"] = {
        "stop_on_duplicate_edits": True,
        "baseline_edits": {},
    }
    write_json(root / "bundle/experiment.json", state["config"])
    state["bundle_hashes"] = hashes(root / "bundle")
    store(root, state)
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    write_json(candidate / "workflow.json", {})
    calls = []

    def duplicate(*a, **kw):
        calls.append(kw)
        return {
            "duplicate_edit": {"matches_proposal": "declared_baseline"},
            "physics_executed": False,
            "verdict": {"task_completed": False},
            "infrastructure_error": None,
        }

    monkeypatch.setattr(campaign, "rollout", duplicate)
    row = campaign.submit(root, candidate)
    assert row["status"] == "duplicate_edit" and len(calls) == 1
    assert calls[0]["baseline_edits"] == {}
    assert campaign.advance(root)["next"] == "stopped"
    assert load(root)["retained_nominal"] is None


@pytest.mark.parametrize("prior", [[], {"": {}}, {"earlier": None}])
def test_prior_edit_registry_rejects_malformed_entries(prior):
    with pytest.raises(ValueError, match="prior_edits"):
        validate_options(
            {
                "backend": {"kind": "reference_refinement"},
                "author_loop": {"stop_on_duplicate_edits": True, "prior_edits": prior},
            }
        )


def test_continuation_checks_imported_edits_before_first_physics(tmp_path, monkeypatch):
    from real2sim2real.agentic_policy.storage import hashes, store

    root = setup(tmp_path)
    state = load(root)
    state["config"]["backend"] = {"kind": "reference_refinement"}
    prior = {
        "reference_times_s": [1.0, 2.0, 3.0],
        "finger_offsets_rad": [[0], [0.02], [0]],
    }
    state["config"]["author_loop"] = {
        "stop_on_duplicate_edits": True,
        "prior_edits": {"earlier": prior},
    }
    write_json(root / "bundle/experiment.json", state["config"])
    state["bundle_hashes"] = hashes(root / "bundle")
    store(root, state)
    backend = fingerprint_backend()
    monkeypatch.setattr(campaign, "make_backend", lambda *a: backend)
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    write_json(candidate / "workflow.json", {})

    def duplicate(*a, **kw):
        assert (
            kw["duplicate_edit_fingerprints"][backend.edit_fingerprint(prior)]
            == "prior:earlier"
        )
        return {
            "duplicate_edit": {"matches_proposal": "prior:earlier"},
            "physics_executed": False,
            "verdict": {"task_completed": False},
            "infrastructure_error": None,
        }

    monkeypatch.setattr(campaign, "rollout", duplicate)
    row = campaign.submit(root, candidate)
    assert row["status"] == "duplicate_edit"
    assert campaign.advance(root)["next"] == "stopped"
    assert load(root)["retained_nominal"] is None
