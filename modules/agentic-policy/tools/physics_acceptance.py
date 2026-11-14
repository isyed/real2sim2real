"""Check an installed backend against an independently frozen state replay."""

import json
import sys
from pathlib import Path

import numpy as np
from real2sim2real.agentic_policy.execution import rollout
from real2sim2real.artifacts import write_json


def main(root):
    root = Path(root)
    assert not Path("/home").exists()
    config = json.loads((root / "bundle/experiment.json").read_text())
    output = root / "rollout"
    output.mkdir()
    result = rollout(root / "bundle", config, root / "candidate", output)
    assert result["infrastructure_error"] is None, result
    assert result["graph"]["graph_status"] == "success", result
    exact = {}
    deltas = {}
    with (
        np.load(root / "expected-interface-physics/states.npz") as a,
        np.load(output / "physics/states.npz") as b,
    ):
        assert set(a.files) == set(b.files)
        for name in a.files:
            exact[name] = bool(np.array_equal(a[name], b[name]))
            deltas[name] = (
                float(np.max(abs(a[name] - b[name])))
                if a[name].shape == b[name].shape
                else None
            )
        samples = len(b["timestamps_s"])
        end = float(b["timestamps_s"][-1])
    receipt = {
        "passed": all(exact.values()),
        "exact": exact,
        "max_abs_deltas": deltas,
        "saved_samples": samples,
        "final_time_s": end,
        "task_completed": result["verdict"]["task_completed"],
        "scope": "Nonzero-motion interface replay, not a generated task policy",
        "workspace_checkouts_visible": False,
        "external_author_calls": False,
    }
    write_json(root / "physics-acceptance.json", receipt)
    print(json.dumps(receipt, indent=2))
    assert receipt["passed"], receipt


if __name__ == "__main__":
    main(sys.argv[1])
