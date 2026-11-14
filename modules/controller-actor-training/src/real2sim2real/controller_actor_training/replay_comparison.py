"""Compare frozen and newly simulated trajectories without feeding them to control."""

import json
from pathlib import Path

import numpy as np

from real2sim2real.artifacts import digest

SEMANTIC_FIELDS = (
    "task_completed",
    "complete_horizon",
    "stop_reason",
    "release",
    "boundary",
    "warnings",
    "hard_failures",
    "strict_quality_pass",
    "control_accepted",
    "records",
)


def compare_replays(expected, actual, *, record_field_aliases=None):
    """Require exact finite arrays and declared diagnostic semantics; no tolerance tuning."""
    expected, actual = Path(expected), Path(actual)
    exact, deltas, shapes = {}, {}, {}
    with (
        np.load(expected / "states.npz", allow_pickle=False) as a,
        np.load(actual / "states.npz", allow_pickle=False) as b,
    ):
        keys = {"qpos", "qvel", "commands", "timestamps_s"}
        if set(a.files) != keys or set(b.files) != keys:
            raise ValueError("Replay requires qpos, qvel, commands and timestamps_s")
        for name in sorted(keys):
            x, y = a[name], b[name]
            finite = bool(np.isfinite(x).all() and np.isfinite(y).all())
            same_shape = x.shape == y.shape
            exact[name] = bool(
                same_shape and finite and x.dtype == y.dtype and np.array_equal(x, y)
            )
            shapes[name] = {"expected": list(x.shape), "actual": list(y.shape)}
            deltas[name] = (
                float(np.max(abs(x - y), initial=0)) if same_shape and finite else None
            )
        times = b["timestamps_s"].copy()
    source = json.loads((expected / "result.json").read_text())
    observed = json.loads((actual / "result.json").read_text())
    aliases = record_field_aliases or {}
    for row in source.get("records", []):
        for old, new in aliases.items():
            if old in row:
                if new in row:
                    raise ValueError("Ambiguous diagnostic field alias")
                row[new] = row.pop(old)
    semantic = {
        key: key in source and key in observed and source[key] == observed[key]
        for key in SEMANTIC_FIELDS
    }
    clock_valid = bool(
        times.ndim == 1
        and len(times) > 1
        and times[0] == 0
        and np.isfinite(times).all()
        and (np.diff(times) > 0).all()
    )
    phase_order = []
    for row in observed.get("records", []):
        if not phase_order or phase_order[-1]["phase"] != row["phase"]:
            phase_order.append({"phase": row["phase"], "first_time_s": row["t"]})
    return {
        "passed": bool(
            clock_valid
            and all(exact.values())
            and all(semantic.values())
            and observed.get("task_completed") is True
            and observed.get("boundary", {}).get("passed") is True
        ),
        "exact": exact,
        "max_abs_deltas": deltas,
        "shapes": shapes,
        "semantic_equality": semantic,
        "record_field_aliases": aliases,
        "clock_valid": clock_valid,
        "saved_samples": len(times),
        "final_time_s": float(times[-1])
        if len(times) and np.isfinite(times[-1])
        else None,
        "phase_order": phase_order,
        "release": observed.get("release"),
        "home": observed.get("boundary"),
        "task_completed": observed.get("task_completed"),
        "control_accepted": observed.get("control_accepted"),
        "baseline_states_sha256": digest(expected / "states.npz"),
        "replay_states_sha256": digest(actual / "states.npz"),
    }
