import json
from pathlib import Path

import pytest
from real2sim2real.agentic_policy import campaign
from real2sim2real.agentic_policy.objectives import assessment, validate
from real2sim2real.agentic_policy.storage import load
from real2sim2real.artifacts import write_json


def bundle(tmp_path, mode="manual"):
    b = tmp_path / "bundle"
    (b / "author").mkdir(parents=True)
    write_json(
        b / "experiment.json",
        {
            "mode": mode,
            "feedback_revisions": 6,
            "instruction": "test",
            "rollout_timeout_s": 10,
            "backend": {"kind": "diagnostic_counter"},
        },
    )
    return b


def test_budget_freeze_and_no_automatic_manual_execution(tmp_path, monkeypatch):
    root = tmp_path / "campaign"
    campaign.initialize(bundle(tmp_path), root)
    p = tmp_path / "proposal"
    p.mkdir()
    write_json(p / "workflow.json", {})
    calls = []

    def evaluate(*args):
        calls.append(1)
        return {"infrastructure_error": None, "verdict": {"task_completed": True}}

    monkeypatch.setattr(campaign, "rollout", evaluate)
    assert campaign.advance(root)["completed_proposals"] == 0
    for _ in range(7):
        campaign.submit(root, p)
    with pytest.raises(ValueError, match="budget"):
        campaign.submit(root, p)
    assert len(calls) == 7
    assert load(root)["retained_nominal"] == "P0"
    (root / "bundle/author/changed").write_text("x")
    with pytest.raises(ValueError, match="changed"):
        campaign.status(root)


def test_interrupted_trial_requires_explicit_retry_of_same_proposal(
    tmp_path, monkeypatch
):
    root = tmp_path / "campaign"
    campaign.initialize(bundle(tmp_path), root)
    p = tmp_path / "proposal"
    p.mkdir()
    write_json(p / "workflow.json", {})

    def fail(*args):
        raise RuntimeError("infrastructure")

    monkeypatch.setattr(campaign, "rollout", fail)
    row = campaign.submit(root, p)
    assert row["status"] == "infrastructure_error"
    with pytest.raises(ValueError, match="retry"):
        campaign.submit(root, p)
    monkeypatch.setattr(
        campaign,
        "rollout",
        lambda *a: {"infrastructure_error": None, "verdict": {"task_completed": False}},
    )
    campaign.retry(root)
    state = load(root)
    assert len(state["rounds"]) == 1 and len(state["rounds"][0]["trials"]) == 2


def test_nominal_first_and_separate_admissions():
    protocol = {
        "target": "control",
        "robustness_cases": [{"name": "light", "backend": {}}],
        "admission": {
            "control": {"limits": {"drift": {"max": 0.01}}},
            "rl": {"require_robustness": False, "limits": {"drift": {"max": 0.03}}},
        },
    }
    validate(protocol)
    r = {"verdict": {"task_completed": False, "drift": 0}}
    assert assessment(r, protocol)["next_objective"] == "nominal_end_to_end"
    r["verdict"].update(task_completed=True, drift=0.02)
    a = assessment(r, protocol)
    assert (
        a["admission"]["rl"]["admitted"] and not a["admission"]["control"]["admitted"]
    )
    assert a["next_objective"] == "robustness_without_nominal_regression"
    r["robustness_cases"] = {
        "light": {"verdict": {"task_completed": True, "drift": 0.02}}
    }
    assert (
        assessment(r, protocol)["next_objective"]
        == "control_acceptance_without_nominal_regression"
    )
    r["verdict"]["drift"] = 0.005
    r["robustness_cases"]["light"]["verdict"]["drift"] = 0.005
    assert assessment(r, protocol)["admission"]["control"]["admitted"]
    del r["verdict"]["drift"]
    assert not assessment(r, protocol)["admission"]["control"]["admitted"]


def test_bundled_graph_runtime_is_private_namespace(tmp_path):
    from real2sim2real.agentic_policy._vendor.gap.runtime.executor import (
        WorkflowExecutor,
    )
    from real2sim2real.agentic_policy._vendor.gap_core.tools import ToolRegistry
    from real2sim2real.agentic_policy._vendor.geometry import tools  # noqa: F401

    registry = ToolRegistry()
    registry.discover_pending()
    graph = {
        "version": 3,
        "nodes": {
            "distance": {
                "type": "tool",
                "tool": "geometry.pose_distance",
                "inputs": {"a": [0.0, 0.0, 0.0], "b": [1.0, 0.0, 0.0]},
            },
            "done": {"type": "end", "status": "success"},
        },
        "edges": [["START", "distance"], ["distance", "done"]],
        "conditional_edges": {},
        "subgraphs": {},
    }
    write_json(tmp_path / "workflow.json", graph)
    executor = WorkflowExecutor(
        tmp_path / "workflow.json",
        tool_registry=registry,
        trace_dir=tmp_path / "trace",
        max_node_workers=1,
        node_visit_cap=10,
        checkpoints="off",
    )
    executor.execute()
    assert executor.exit_status == "success"
    assert (
        registry.invoke("geometry.pose_distance", a=[0, 0, 0], b=[1, 0, 0])["distance"]
        == 1.0
    )


def test_nominal_success_is_retained_when_later_revision_regresses(
    tmp_path, monkeypatch
):
    root = tmp_path / "campaign"
    campaign.initialize(bundle(tmp_path), root)
    p = tmp_path / "proposal"
    p.mkdir()
    write_json(p / "workflow.json", {})
    outcomes = iter([True, False])
    monkeypatch.setattr(
        campaign,
        "rollout",
        lambda *a: {
            "infrastructure_error": None,
            "verdict": {"task_completed": next(outcomes)},
        },
    )
    campaign.submit(root, p)
    campaign.submit(root, p)
    assert load(root)["retained_nominal"] == "P0"


def test_export_is_common_artifact_without_training_admission(tmp_path, monkeypatch):
    from real2sim2real.artifacts import load as load_artifact

    root = tmp_path / "campaign"
    campaign.initialize(bundle(tmp_path), root)
    p = tmp_path / "proposal"
    p.mkdir()
    write_json(p / "workflow.json", {})
    monkeypatch.setattr(
        campaign,
        "rollout",
        lambda *a: {"infrastructure_error": None, "verdict": {"task_completed": True}},
    )
    campaign.submit(root, p)
    campaign.export_policy(root, "P0", tmp_path / "export")
    result = load_artifact(tmp_path / "export", kinds={"controller-policy-candidate"})
    assert not result["metadata"]["training_admitted"]


def test_vendored_sources_have_pinned_content():
    import hashlib

    from real2sim2real.agentic_policy import _vendor

    root = Path(_vendor.__file__).parent
    provenance = json.loads((root / "provenance.json").read_text())
    for name, component in provenance.items():
        assert len(component["revision"]) == 40
        for relative, hashes in component["files"].items():
            assert (
                hashlib.sha256((root / name / relative).read_bytes()).hexdigest()
                == hashes["bundled_sha256"]
            )


def test_case_infrastructure_failure_does_not_erase_nominal_success(
    tmp_path, monkeypatch
):
    source = bundle(tmp_path)
    config = json.loads((source / "experiment.json").read_text())
    config["evaluation"] = {
        "robustness_cases": [{"name": "missing-case", "backend": {"kind": "missing"}}]
    }
    write_json(source / "experiment.json", config)
    root = tmp_path / "campaign"
    campaign.initialize(source, root)
    p = tmp_path / "proposal"
    p.mkdir()
    write_json(p / "workflow.json", {})

    def evaluate(bundle, config, candidate, output):
        if config["backend"]["kind"] == "missing":
            raise RuntimeError("case cannot start")
        return {"infrastructure_error": None, "verdict": {"task_completed": True}}

    monkeypatch.setattr(campaign, "rollout", evaluate)
    row = campaign.submit(root, p)
    assert row["result"]["assessment"]["nominal_completed"]
    assert not row["result"]["assessment"]["robustness_passed"]
    assert load(root)["retained_nominal"] == "P0"


def test_author_catalog_includes_all_bundled_tools_and_source():
    from real2sim2real.agentic_policy.catalog import author_catalog

    tools, access = author_catalog(
        {"robot.observe": {"inputs": {}, "description": "observe"}}
    )
    assert len(tools) == 23
    assert (
        "geometry.pose_distance" in tools and "geometry.compute_feature_mate" in tools
    )
    assert access["python_sources"]
    assert all(Path(p).is_file() for p in access["python_sources"])
    assert any(p.endswith("continuous_policy.py") for p in access["python_sources"])
