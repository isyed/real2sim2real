"""Strict GaP pass-through phases over one persistent, zero-correction replay.

GaP is an optional dependency, loaded only when executing the graph. This module
does not discover skill registries, import policy scripts or generate policies.
"""

import json
import threading
from copy import deepcopy
from pathlib import Path

import numpy as np

TOOL = "controller.advance_frozen_phase"


def phase_order(records):
    """Recover a linear phase sequence; revisits require a different contract."""
    phases = []
    for record in records:
        phase = record["phase"]
        if not isinstance(phase, str) or not phase:
            raise ValueError("Nonempty phase names required")
        if not phases or phase != phases[-1]:
            if phase in phases:
                raise ValueError("Pass-through phases cannot revisit an earlier phase")
            phases.append(phase)
    if not phases:
        raise ValueError("Nonempty phase sequence required")
    return tuple(phases)


def workflow(phases):
    """GaP v3 tool-only graph; completion is distinct from physical success."""
    phases = phase_order([{"phase": p} for p in phases])
    names = [f"phase_{i:02d}" for i in range(len(phases))]
    return {
        "version": 3,
        "nodes": {
            **{
                name: {"type": "tool", "tool": TOOL, "inputs": {"phase": phase}}
                for name, phase in zip(names, phases, strict=True)
            },
            "complete": {"type": "end", "status": "success"},
        },
        "edges": [list(pair) for pair in zip(["START", *names], [*names, "complete"])],
        "conditional_edges": {},
        "subgraphs": {},
    }


def validate_workflow(document, phases):
    # A reproduction run admits exactly the generated graph, not arbitrary code
    # or extra waits/resets/tools. Candidate policy search is a separate stage.
    if document != workflow(phases):
        raise ValueError("Only the complete frozen phase graph is admitted")


class ZeroPhaseSession:
    """Sequential graph calls share one environment and only send zero actions."""

    def __init__(self, environment_factory, phases):
        self.factory = environment_factory
        self.phases = phase_order([{"phase": p} for p in phases])
        self.env = None
        self.reset_count = 0
        self.steps = 0
        self.info = None
        self.finished = False
        self.failed = False
        self.spans = []
        self._lock = threading.Lock()

    def advance(self, phase: str) -> dict:
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Concurrent graph calls cannot advance shared physics")
        try:
            if self.failed or self.finished or len(self.spans) >= len(self.phases):
                raise RuntimeError("The graph session has ended")
            if phase != self.phases[len(self.spans)]:
                raise ValueError("Graph phases must execute exactly once in order")
            if self.env is None:
                self.env = self.factory()
                _, self.info = self.env.reset()
                self.reset_count += 1
            if self.info["phase"] != phase:
                raise ValueError("Replay observation does not match the graph phase")
            start = float(self.info["time_s"])
            previous_steps = self.steps
            if self.spans and (
                self.spans[-1]["end_time_s"] != start
                or self.spans[-1]["end_step"] != previous_steps
            ):
                raise RuntimeError("Graph boundary advanced or reset simulation")
            stages = set()
            while not self.finished and self.info["phase"] == phase:
                stages.add(self.info.get("task_stage"))
                before = float(self.info["time_s"])
                _, _, terminated, truncated, info = self.env.step(
                    np.zeros(self.env.action_dim, dtype=float)
                )
                if truncated:
                    raise RuntimeError("Truncated graph replay cannot establish parity")
                if not np.isfinite(info["time_s"]) or info["time_s"] <= before:
                    raise RuntimeError("Replay clock did not advance monotonically")
                self.info = info
                self.steps += 1
                self.finished = bool(terminated)
            if self.finished != (len(self.spans) + 1 == len(self.phases)):
                raise RuntimeError("Replay ended before or after the final graph phase")
            span = {
                "phase": phase,
                "start_time_s": start,
                "end_time_s": float(self.info["time_s"]),
                "start_step": previous_steps,
                "end_step": self.steps,
                "policy_steps": self.steps - previous_steps,
                "actual_task_stages": sorted(s for s in stages if s is not None),
                "next_phase": None if self.finished else self.info["phase"],
                "terminated": self.finished,
            }
            self.spans.append(span)
            return deepcopy(span)
        except BaseException:
            self.failed = True
            raise
        finally:
            self._lock.release()

    def receipt(self):
        boundaries = [
            {
                "from_phase": a["phase"],
                "to_phase": b["phase"],
                "simulation_gap_s": b["start_time_s"] - a["end_time_s"],
                "extra_policy_steps": b["start_step"] - a["end_step"],
            }
            for a, b in zip(self.spans, self.spans[1:])
        ]
        return {
            "phases": list(self.phases),
            "spans": deepcopy(self.spans),
            "boundaries": boundaries,
            "reset_count": self.reset_count,
            "policy_steps": self.steps,
            "complete": self.finished
            and not self.failed
            and len(self.spans) == len(self.phases),
            "boundary_clock_preserved": all(
                b["simulation_gap_s"] == 0 and b["extra_policy_steps"] == 0
                for b in boundaries
            ),
        }

    def close(self):
        if self.env is not None:
            self.env.close()


def execute(graph_path, session, trace_dir):
    """Use the actual upstream scheduler with one explicitly registered tool."""
    validate_workflow(json.loads(Path(graph_path).read_text()), session.phases)
    from gap.runtime.executor import WorkflowExecutor
    from gap_core.tools import ToolRegistry

    registry = ToolRegistry()
    registry.register_callable(
        TOOL,
        session.advance,
        summary="Advance the unchanged controller through one scheduled phase",
        requires_ctx=False,
    )
    executor = WorkflowExecutor(
        graph_path,
        tool_registry=registry,
        trace_dir=trace_dir,
        max_node_workers=1,
        node_visit_cap=len(session.phases) + 2,
        checkpoints="off",
    )
    executor.execute()
    if executor.exit_status != "success" or not session.receipt()["complete"]:
        raise RuntimeError("Graph did not complete the entire frozen replay")
    return executor.exit_status
