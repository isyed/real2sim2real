import json
from copy import deepcopy

import pytest
from real2sim2real.controller_actor_training.seating_graph_replay import (
    FrozenSeatingSession,
    execute_zero_seating_graph,
)


def seed():
    return {
        "version": 3,
        "nodes": {
            "configure": {
                "type": "tool",
                "tool": "controller.configure_seating",
                "inputs": {
                    "pad_offsets_object_m": {"thumb": [0.0, 0.0, 0.0]},
                    "closure_offsets_rad": {},
                },
            },
            "sequence": {
                "type": "tool",
                "tool": "controller.run_frozen_sequence",
                "inputs": {},
            },
            "complete": {"type": "end", "status": "success"},
        },
        "edges": [
            ["START", "configure"],
            ["configure", "sequence"],
            ["sequence", "complete"],
        ],
        "conditional_edges": {},
        "subgraphs": {},
    }


def test_configuration_does_not_start_physics_and_run_is_once():
    calls = []
    session = FrozenSeatingSession(lambda: calls.append(1) or {"task_completed": True})
    session.configure({"thumb": [0, 0, 0]}, {})
    assert calls == []
    session.run()
    assert calls == [1] and session.receipt()["complete"]
    with pytest.raises(RuntimeError):
        session.run()
    assert calls == [1]


def test_nonzero_and_out_of_order_calls_cannot_start_physics():
    for action in [
        lambda s: s.configure({"thumb": [0.001, 0, 0]}, {}),
        lambda s: s.run(),
    ]:
        session = FrozenSeatingSession(lambda: pytest.fail("Physics must not start"))
        with pytest.raises((ValueError, RuntimeError)):
            action(session)
        assert not session.receipt()["complete"]


def test_real_scheduler_preserves_task_failure_and_rejects_changed_graph(tmp_path):
    pytest.importorskip("gap")
    graph = seed()
    path = tmp_path / "workflow.json"
    path.write_text(json.dumps(graph))
    session = FrozenSeatingSession(lambda: {"task_completed": False})
    result = execute_zero_seating_graph(
        path,
        graph,
        pads=["thumb"],
        joints=[],
        session=session,
        trace_dir=tmp_path / "trace",
    )
    assert result["graph_status"] == "success"
    assert result["physical_task_completed"] is False
    changed = deepcopy(graph)
    changed["nodes"]["configure"]["inputs"]["pad_offsets_object_m"]["thumb"][0] = 0.001
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError):
        execute_zero_seating_graph(
            path,
            graph,
            pads=["thumb"],
            joints=[],
            session=FrozenSeatingSession(lambda: pytest.fail("No run")),
            trace_dir=tmp_path / "rejected",
        )
