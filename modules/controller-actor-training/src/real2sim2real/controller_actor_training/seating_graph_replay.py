"""GaP zero-edit seating graph around one complete downstream controller run.

The downstream callback owns physics, assets, deadlines and independent task
evaluation. The graph cannot reset, pause or change that run's internal phases.
Nonzero proposals require a separate candidate execution adapter.
"""

import json
import threading
from copy import deepcopy
from pathlib import Path

from .seating_contract import validate_seating_proposal


class FrozenSeatingSession:
    def __init__(self, run_sequence):
        self.run_sequence = run_sequence
        self.configured = self.failed = self.finished = False
        self.configure_calls = self.sequence_calls = 0
        self.result = None
        self._lock = threading.Lock()

    def configure(self, pad_offsets_object_m: dict, closure_offsets_rad: dict) -> dict:
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Concurrent calls cannot alter a frozen session")
        try:
            if self.configured or self.failed or self.finished:
                raise RuntimeError("Configure must execute exactly once before physics")
            values = [x for v in pad_offsets_object_m.values() for x in v]
            values += list(closure_offsets_rad.values())
            if any(type(x) not in (int, float) or x != 0 for x in values):
                raise ValueError("Replay admits zero edits only; fitting is bypassed")
            self.configured = True
            self.configure_calls += 1
            return {"zero_edits": True, "simulation_steps": 0, "fitting_calls": 0}
        except BaseException:
            self.failed = True
            raise
        finally:
            self._lock.release()

    def run(self) -> dict:
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Concurrent graph calls cannot start another simulation")
        try:
            if not self.configured or self.failed or self.finished:
                raise RuntimeError("One configured, continuous sequence required")
            self.sequence_calls += 1
            result = self.run_sequence()
            if (
                not isinstance(result, dict)
                or type(result.get("task_completed")) is not bool
            ):
                raise ValueError("Independent physical task verdict required")
            self.result = deepcopy(result)
            self.finished = True
            return deepcopy(result)
        except BaseException:
            self.failed = True
            raise
        finally:
            self._lock.release()

    def receipt(self):
        return {
            "configure_calls": self.configure_calls,
            "sequence_calls": self.sequence_calls,
            "complete": self.finished and not self.failed,
            "physical_task_completed": self.result.get("task_completed")
            if self.result
            else None,
            "fitting_bypassed": True,
            "internal_phase_handoffs": "owned by the continuous controller",
        }


def execute_zero_seating_graph(graph_path, seed, *, pads, joints, session, trace_dir):
    """Execute the actual optional GaP scheduler; reject edits before tool calls."""
    document = json.loads(Path(graph_path).read_text())
    validate_seating_proposal(document, seed, pads=pads, joints=joints)
    if document != seed:
        raise ValueError("Zero replay requires the exact frozen seed graph")
    from gap.runtime.executor import WorkflowExecutor
    from gap_core.tools import ToolRegistry

    registry = ToolRegistry()
    registry.register_callable(
        "controller.configure_seating",
        session.configure,
        summary="Configure zero edits without stepping physics",
        requires_ctx=False,
    )
    registry.register_callable(
        "controller.run_frozen_sequence",
        session.run,
        summary="Run one complete continuous frozen controller",
        requires_ctx=False,
    )
    executor = WorkflowExecutor(
        graph_path,
        tool_registry=registry,
        trace_dir=trace_dir,
        max_node_workers=1,
        node_visit_cap=4,
        checkpoints="off",
    )
    executor.execute()
    receipt = session.receipt()
    if executor.exit_status != "success" or not receipt["complete"]:
        raise RuntimeError("Graph did not finish the continuous replay")
    return {"graph_status": executor.exit_status, **receipt}
