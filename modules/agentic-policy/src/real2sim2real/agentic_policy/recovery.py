"""Recover reporting after a trusted simulation clock rejection, never a pass."""

import json
from pathlib import Path

import numpy as np

from real2sim2real.artifacts import digest, write_json

from .objectives import assessment
from .storage import load, locked, store


def record_clock_rejection(root):
    """Classify an already stopped unstable rollout without editing/replaying it.

    A clock reset can make the common posthoc audit raise before the campaign
    records the simulator's rejection. Require both trusted fault evidence and
    nonmonotonic saved time; unrelated infrastructure failures remain errors.
    Invalid trajectories receive no normal completion/retention metrics.
    """
    root = Path(root).resolve()
    with locked(root):
        state = load(root)
        if (
            not state["rounds"]
            or state["rounds"][-1]["status"] != "infrastructure_error"
        ):
            raise ValueError("An interrupted reporting stage is required")
        row = state["rounds"][-1]
        trial = root / row["trials"][-1]
        previous = row["result"]
        if (
            previous.get("infrastructure_error")
            != "ValueError: Finite matching sampled states with a strictly increasing clock required"
        ):
            raise ValueError(
                "This recovery applies only to the saved-clock audit error"
            )
        status = json.loads((trial / "physics/status.json").read_text())
        graph = json.loads((trial / "work/graph-result.json").read_text())
        if (
            status.get("fault") != "simulation_clock_reset_or_invalid_step"
            or graph.get("graph_status") != "failure"
        ):
            raise ValueError(
                "Trusted simulator clock rejection and graph failure required"
            )
        source = trial / "physics/states.npz"
        with np.load(source, allow_pickle=False) as arrays:
            times = arrays["timestamps_s"].copy()
        resets = np.flatnonzero(np.diff(times) <= 0)
        if len(times) < 2 or not np.isfinite(times).all() or not len(resets):
            raise ValueError("Saved timestamps must demonstrate the reported reset")
        end = int(resets[0])
        observations = json.loads((trial / "physics/observations.json").read_text())
        if len(observations) != len(times):
            raise ValueError(
                "Saved observations must match the rejected state sequence"
            )
        evidence = {
            "fault": status["fault"],
            "last_sample_before_reset_s": float(times[end]),
            "reset_sample_s": float(times[end + 1]),
            "last_valid_observations": observations[max(0, end - 2) : end + 1],
            "normal_audit_metrics_available": False,
            "original_states_sha256": digest(source),
            "physics_rerun": False,
            "original_reporting_error": previous,
        }
        receipt = {
            "graph": graph,
            "infrastructure_error": None,
            "execution_rejection": status["fault"],
            "numerical_rejection": evidence,
            "tool_calls": None,
            "elapsed_s": None,
            "verdict": {
                "task_completed": False,
                "interface_task_completed": False,
                "diagnostics": {},
            },
        }
        receipt["assessment"] = assessment(
            receipt, state["config"].get("evaluation", {})
        )
        write_json(trial / "reporting-recovery.json", evidence)
        write_json(trial / "result.json", receipt)
        row.update(status="evaluated", result=receipt)
        store(root, state)
        load(root)
        return receipt
