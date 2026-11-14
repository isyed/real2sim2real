"""Replay a frozen manual reference using installed wheels and isolated GaP RPC.

Only the bundle and generated output belong in this process's sandbox. Expected
rollout states are compared afterward, outside the simulation process.
"""

import json
import sys
from pathlib import Path

from real2sim2real.agentic_policy.execution import rollout
from real2sim2real.agentic_policy.storage import hashes

from real2sim2real.artifacts import write_json


def main(root):
    root = Path(root)
    assert not Path("/home").exists(), "Workspace checkouts must be hidden"
    config = json.loads((root / "bundle/experiment.json").read_text())
    assert config["backend"]["kind"] == "frozen_reference"
    pins = hashes(root / "bundle")
    candidate = root / "candidate"
    candidate.mkdir()
    write_json(
        candidate / "workflow.json",
        {
            "version": 3,
            "nodes": {
                "sequence": {
                    "type": "tool",
                    "tool": "controller.run_reference",
                    "inputs": {},
                },
                "done": {"type": "end", "status": "success"},
            },
            "edges": [["START", "sequence"], ["sequence", "done"]],
            "conditional_edges": {},
            "subgraphs": {},
        },
    )
    output = root / "rollout"
    output.mkdir()
    result = rollout(root / "bundle", config, candidate, output)
    unchanged = pins == hashes(root / "bundle")
    receipt = {
        "graph_status": result["graph"],
        "infrastructure_error": result["infrastructure_error"],
        "execution_rejection": result["execution_rejection"],
        "tool_calls": result["tool_calls"],
        "verdict": result["verdict"],
        "inputs_unchanged": unchanged,
        "workspace_checkouts_visible": False,
        "external_author_calls": False,
        "scope": "Fresh closed-loop manual reference physics; expected rollout states are not inputs.",
    }
    write_json(root / "replay-receipt.json", receipt)
    print(json.dumps(receipt, indent=2))
    assert unchanged and result["infrastructure_error"] is None
    assert result["graph"]["graph_status"] == "success"
    assert (
        result["verdict"]["task_completed"] and result["verdict"]["boundary"]["passed"]
    )


if __name__ == "__main__":
    main(sys.argv[1])
