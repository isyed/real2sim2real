"""Execute one declarative proposal through GaP over a persistent native replay."""

import argparse
import json
import math
import os
import sys
import threading
from copy import deepcopy
from pathlib import Path

from real2sim2real.artifacts import digest, write_json

from .passthrough_graph import TOOL, phase_order, workflow
from .proposal_contract import CONFIGURE_TOOL, seeded_workflow, validate_proposal


class NativePhaseSession:
    """Phase boundaries may resume the simulator, never inject actions or time."""

    def __init__(self, factory, phases, parameters):
        self.factory, self.phases = factory, tuple(phases)
        self.parameters = deepcopy(parameters)
        self.configured = self.failed = self.finished = False
        self.worker = self.event = self.result = None
        self.spans = []
        self.reset_count = 0
        self._lock = threading.Lock()

    def configure(
        self,
        vectors_object_m: dict,
        rotation_vectors_object_rad: dict,
        finger_joint_offsets_rad: dict,
    ) -> dict:
        if self.configured or self.worker is not None:
            raise RuntimeError("Proposal configuration is immutable after entry")
        received = {
            "vectors_object_m": vectors_object_m,
            "rotation_vectors_object_rad": rotation_vectors_object_rad,
            "finger_joint_offsets_rad": finger_joint_offsets_rad,
        }
        if received != self.parameters:
            raise ValueError("Runtime parameters differ from the admitted proposal")
        self.configured = True
        return {"configured": True, "simulation_steps": 0}

    def advance(self, phase: str) -> dict:
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Concurrent phase execution forbidden")
        try:
            if self.failed or self.finished or not self.configured:
                raise RuntimeError("Session is not available")
            if phase != self.phases[len(self.spans)]:
                raise ValueError("Phases must execute once in order")
            if self.worker is None:
                self.worker = self.factory()
                self.reset_count += 1
                self.event = next(self.worker)
            if self.event["phase"] != phase:
                raise ValueError("Native phase differs from graph phase")
            start = deepcopy(self.event)
            try:
                self.event = next(self.worker)
            except StopIteration as done:
                self.result = done.value
                if "incomplete_pilot" in self.result.get("rejected_for", []):
                    raise RuntimeError("Native replay returned an incomplete pilot")
                self.finished = True
                self.event = {
                    "phase": None,
                    "time_s": self.result["records"][-1]["time_s"],
                }
            if (
                not math.isfinite(start["time_s"])
                or not math.isfinite(self.event["time_s"])
                or self.event["time_s"] <= start["time_s"]
            ):
                raise RuntimeError("Simulation clock did not advance")
            if self.finished != (len(self.spans) + 1 == len(self.phases)):
                raise RuntimeError("Native replay ended outside the final phase")
            span = {
                "phase": phase,
                "start_time_s": start["time_s"],
                "end_time_s": self.event["time_s"],
                "next_phase": self.event["phase"],
            }
            if self.spans and self.spans[-1]["end_time_s"] != span["start_time_s"]:
                raise RuntimeError("Graph inserted a clock gap")
            self.spans.append(span)
            return deepcopy(span)
        except BaseException:
            self.failed = True
            raise
        finally:
            self._lock.release()

    def receipt(self):
        return {
            "complete": self.finished and not self.failed,
            "reset_count": self.reset_count,
            "spans": deepcopy(self.spans),
            "boundaries": [
                {"simulation_gap_s": b["start_time_s"] - a["end_time_s"]}
                for a, b in zip(self.spans, self.spans[1:])
            ],
        }

    def close(self):
        if self.worker is not None:
            self.worker.close()


def validate_config(config):
    """Check the run contract before importing optional simulator packages."""
    if set(config) != {
        "baseline",
        "workflow",
        "allowed_joints",
        "output",
        "deadline_s",
    }:
        raise ValueError(
            "Only pinned baseline/graph, joint allowlist and output/deadline allowed"
        )
    deadline = config["deadline_s"]
    if type(deadline) not in (int, float) or not 0 < deadline <= 900:
        raise ValueError("Finite wall deadline up to 900 seconds required")
    return deadline


def run(config):
    """Execute in-process; callers must supply an external wall-clock bound.

    The CLI provides that process deadline automatically.
    """
    validate_config(config)
    from gap.runtime.executor import WorkflowExecutor
    from gap_core.tools import ToolRegistry

    from .arm_replay_pilot import rollout
    from .bounded_grasp_correction import validate
    from .paired_acquisition_audit import pinned_json
    from .passthrough_graph_validate import package_receipt, source_files

    baseline = pinned_json(config["baseline"])
    graph = pinned_json(config["workflow"])
    phases = phase_order(baseline["records"])
    parameters = validate_proposal(
        graph, seeded_workflow(workflow(phases)), config["allowed_joints"]
    )
    replay_config = deepcopy(baseline["config"])
    out = Path(config["output"])
    replay_config["output"] = str(out / "replay")
    replay_config["purpose"] = (
        "One declared bounded graph proposal over a frozen continuous controller; "
        "unchanged scene, physical settings, native reference timing and acceptance."
    )
    replay_config["bounded_grasp_correction"] = {
        "baseline": config["baseline"],
        **parameters,
    }
    validate(replay_config["bounded_grasp_correction"], replay_config)
    pinned = source_files(config["baseline"]["path"])
    out.mkdir(parents=True, exist_ok=False)
    graph_path = out / "workflow.json"
    graph_path.write_bytes(Path(config["workflow"]["path"]).read_bytes())
    snapshot = out / "implementation"
    snapshot.mkdir()
    implementation_hashes = {}
    for name in ("bounded_policy_graph.py", "proposal_contract.py"):
        target = snapshot / name
        target.write_bytes(Path(__file__).with_name(name).read_bytes())
        implementation_hashes[name] = digest(target)
    write_json(
        out / "manifest.json",
        {
            "schema": "bounded-policy-graph/1",
            "config": config,
            "source_files": pinned,
            "controller_config": replay_config,
            "graph_sha256": digest(graph_path),
            "parameters": parameters,
            "gap": package_receipt("graph-as-policy"),
            "implementation_sha256": implementation_hashes,
            "training_run": False,
            "promotion_eligible": False,
        },
    )
    session = NativePhaseSession(
        lambda: rollout(replay_config, phase_handoff=True), phases, parameters
    )
    try:
        registry = ToolRegistry()
        registry.register_callable(
            CONFIGURE_TOOL,
            session.configure,
            summary="Bind the admitted bounded correction once",
            requires_ctx=False,
        )
        registry.register_callable(
            TOOL,
            session.advance,
            summary="Advance one native continuous phase",
            requires_ctx=False,
        )
        executor = WorkflowExecutor(
            graph_path,
            tool_registry=registry,
            trace_dir=out / "graph-trace",
            max_node_workers=1,
            node_visit_cap=len(phases) + 3,
            checkpoints="off",
        )
        executor.execute()
        if executor.exit_status != "success" or not session.receipt()["complete"]:
            raise RuntimeError(
                "Proposal did not complete the entire continuous protocol"
            )
        if pinned != source_files(config["baseline"]["path"]):
            raise ValueError("Frozen inputs changed during execution")
        if digest(graph_path) != config["workflow"]["sha256"]:
            raise ValueError("Proposal changed during execution")
        receipt = {
            "graph": session.receipt(),
            "graph_status": executor.exit_status,
            "task_success": session.result["success"],
            "rejected_for": session.result["rejected_for"],
            "result_sha256": digest(out / "replay/result.json"),
            "pinned_sources_unchanged": True,
            "promotion_eligible": False,
        }
        write_json(out / "receipt.json", receipt)
        return receipt
    except BaseException as error:
        write_json(
            out / "interrupted.json", {"reason": str(error), "graph": session.receipt()}
        )
        raise
    finally:
        session.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    deadline = validate_config(config)
    if args.worker:
        run(config)
    else:
        from real2sim2real.bounded_process import run as bounded

        return bounded(
            [
                sys.executable,
                "-m",
                __spec__.name,
                "--config",
                str(args.config.resolve()),
                "--worker",
            ],
            timeout=deadline,
            env={
                **os.environ,
                "CUDA_VISIBLE_DEVICES": "",
                "OPENBLAS_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
            },
        )


if __name__ == "__main__":
    raise SystemExit(main())
