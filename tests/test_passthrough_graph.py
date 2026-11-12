import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from real2sim2real.controller_actor_training.passthrough_graph import (
    ZeroPhaseSession,
    execute,
    phase_order,
    validate_workflow,
    workflow,
)
from real2sim2real.controller_actor_training.passthrough_graph_validate import (
    environment_config,
    source_files,
)

from real2sim2real.artifacts import digest


class Replay:
    action_dim = 4

    def __init__(self):
        self.phases = ("approach", "close", "return", "release")
        self.i = 0
        self.actions = []
        self.resets = 0
        self.closed = False
        self.truncate = False
        self.freeze = False

    def info(self):
        return {"time_s": self.i * 0.02, "phase": self.phases[min(self.i // 2, 3)]}

    def reset(self):
        self.resets += 1
        return np.zeros(1), self.info()

    def step(self, action):
        self.actions.append(action.copy())
        if not self.freeze:
            self.i += 1
        return np.zeros(1), 0.0, self.i == 8, self.truncate, self.info()

    def close(self):
        self.closed = True


def test_phase_handoffs_keep_one_session_zero_actions_and_no_clock_gap():
    replay = Replay()
    session = ZeroPhaseSession(lambda: replay, replay.phases)
    for phase in replay.phases:
        result = session.advance(phase)
        assert result["policy_steps"] == 2
        result["end_time_s"] = -10  # External output cannot corrupt the next node.
    receipt = session.receipt()
    assert receipt["complete"] and receipt["boundary_clock_preserved"]
    assert receipt["reset_count"] == replay.resets == 1
    assert receipt["policy_steps"] == 8
    assert np.array_equal(replay.actions, np.zeros((8, 4)))
    assert [s["start_step"] for s in receipt["spans"]] == [0, 2, 4, 6]
    assert all(b["simulation_gap_s"] == 0 for b in receipt["boundaries"])
    session.close()
    assert replay.closed
    with pytest.raises(RuntimeError):
        session.advance("release")


@pytest.mark.parametrize("mode", ["truncate", "freeze", "early_end"])
def test_incomplete_or_stalled_replay_cannot_claim_completion(mode):
    replay = Replay()
    if mode == "early_end":
        phases = (*replay.phases, "extra")
    else:
        setattr(replay, mode, True)
        phases = replay.phases
    session = ZeroPhaseSession(lambda: replay, phases)
    with pytest.raises(RuntimeError):
        for phase in phases:
            session.advance(phase)
    assert session.receipt()["complete"] is False


def test_out_of_order_call_fails_before_any_environment_initialization():
    def forbidden():
        raise AssertionError("No initialization on an invalid first phase")

    session = ZeroPhaseSession(forbidden, ["approach", "close"])
    with pytest.raises(ValueError, match="in order"):
        session.advance("close")
    assert session.reset_count == 0


def test_concurrent_calls_cannot_advance_the_same_physics():
    replay = Replay()
    session = ZeroPhaseSession(lambda: replay, replay.phases)
    session._lock.acquire()
    try:
        with pytest.raises(RuntimeError, match="Concurrent"):
            session.advance("approach")
    finally:
        session._lock.release()
    assert replay.resets == 0


def test_detects_unexpected_state_advancement_between_graph_nodes():
    replay = Replay()
    session = ZeroPhaseSession(lambda: replay, replay.phases)
    session.advance("approach")
    session.info["time_s"] += 0.02
    with pytest.raises(RuntimeError, match="boundary"):
        session.advance("close")


@pytest.mark.parametrize("kind", ["extra_tool", "skip", "reset", "script", "parallel"])
def test_graph_contract_rejects_changes_that_can_break_parity(kind):
    phases = Replay().phases
    doc = workflow(phases)
    if kind == "extra_tool":
        doc["nodes"]["wait"] = {"type": "tool", "tool": "robot.wait_steps"}
    elif kind == "skip":
        doc["edges"][0][1] = "phase_01"
    elif kind == "reset":
        doc["nodes"]["phase_01"]["tool"] = "controller.reset"
    elif kind == "script":
        doc["nodes"]["phase_00"] = {"type": "script", "script": "unsafe.py"}
    else:
        doc["edges"].append(["START", "phase_01"])
    with pytest.raises(ValueError, match="Only the complete"):
        validate_workflow(doc, phases)


def test_phase_sequence_rejects_empty_invalid_and_revisited_phases():
    for records in (
        [],
        [{"phase": None}],
        [{"phase": ""}],
        [{"phase": "a"}, {"phase": "b"}, {"phase": "a"}],
    ):
        with pytest.raises(ValueError):
            phase_order(records)
    assert phase_order([{"phase": "a"}, {"phase": "a"}]) == ("a",)


def test_frozen_configuration_cannot_take_control_overrides(tmp_path):
    config = {
        "baseline_result": "baseline.json",
        "baseline_sha256": "a" * 64,
        "output": str(tmp_path),
        "deadline_s": 30,
        "policy_period_s": 0.02,
        "workflow": "workflow.json",
        "workflow_sha256": "b" * 64,
    }
    original = deepcopy(config)
    env = environment_config(config)
    assert config == original
    assert Path(env["output"]).name == "replay"
    for bad in (
        dict(config, timestep_s=0.001),
        dict(config, deadline_s=901),
        dict(config, deadline_s=float("nan")),
        dict(config, deadline_s=True),
    ):
        with pytest.raises(ValueError):
            environment_config(bad)


def test_evidence_chain_detects_nested_geometry_changes(tmp_path):
    scene = tmp_path / "scene.xml"
    scene.write_text("<mujoco/>")
    states = tmp_path / "states.npz"
    states.write_bytes(b"state receipt")
    geometry = tmp_path / "object.obj"
    geometry.write_text("v 0 0 0")
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps({"sources": {str(geometry): digest(geometry)}}))
    result = tmp_path / "result.json"
    result.write_text(
        json.dumps(
            {
                "scene_sha256": digest(scene),
                "states_sha256": digest(states),
                "sources": {str(audit): digest(audit)},
            }
        )
    )
    assert len(source_files(result)) == 5
    geometry.write_text("v 1 0 0")
    with pytest.raises(ValueError, match="Pinned input changed"):
        source_files(result)


def test_upstream_gap_scheduler_executes_only_registered_zero_phase_tool(tmp_path):
    pytest.importorskip("gap.runtime.executor")
    replay = Replay()
    session = ZeroPhaseSession(lambda: replay, replay.phases)
    graph = tmp_path / "workflow.json"
    graph.write_text(json.dumps(workflow(replay.phases)))
    assert execute(graph, session, tmp_path / "trace") == "success"
    assert session.receipt()["complete"]
    assert replay.resets == 1 and len(replay.actions) == 8
    assert np.array_equal(replay.actions, np.zeros((8, 4)))
    assert (tmp_path / "trace" / "dag_trace.json").is_file()
    session.close()
