"""Installed-wheel graph/RPC smoke test against a supplied portable robot scene.

Uses the initial posture and existing pad locations, not a generated task policy.
The scene must start supported at acquisition, away from the final placement.
The output intentionally fails the overall manipulation task. CPU MuJoCo only.
"""

import json
import sys
from pathlib import Path

from real2sim2real.agentic_policy.execution import rollout

from real2sim2real.artifacts import write_json


def main(root):
    root = Path(root)
    assert not Path("/home").exists(), "Run with checkouts hidden"
    config = json.loads((root / "bundle/experiment.json").read_text())
    candidate = root / "skill-candidate"
    candidate.mkdir()
    (candidate / "policy.py").write_text("""import json
from pathlib import Path
from typing import TypedDict
import numpy as np
from real2sim2real.agentic_policy.api import NodeContext
from real2sim2real.agentic_policy.manipulation import pose

class Result(TypedDict):
    checked: bool

def run(ctx: NodeContext) -> Result:
    receipts = {}
    def call(name, **inputs):
        value = ctx.tool('robot.' + name, **inputs)
        receipts.setdefault(name, []).append(value)
        Path('skill-checks.json').write_text(json.dumps(receipts, indent=2))
        return value
    call('describe_skills')
    start = call('observe')
    q = start['commanded_joints_rad']
    fit = call('fit_grasp', wrist_pose_world=pose(start, 'wrist').tolist(),
               seed_joints_rad=start['joints_rad'], pad_indices=list(range(len(start['pad_forces_n']))),
               targets_world_m=start['pad_points_world_m'], inward_normals_world=start['pad_normals_world'])
    assert fit['fit_pass'] and not fit['loaded_hold_verified']
    assert call('observe')['time_s']==start['time_s']
    acquisition = call('acquire_grasp', positions=[q], durations_s=[.1], pressure_targets_n=[0.] * len(start['pad_forces_n']))
    assert acquisition['executed'] and not acquisition['loaded_hold_verified']
    current = call('observe')
    plan = call('plan_transport', object_poses_world=[pose(current, 'object').tolist()], durations_s=[.1])
    assert plan['admitted'], plan
    assert call('observe')['time_s']==current['time_s']
    assert call('execute_transport', plan_id=plan['plan_id'])['executed']
    assert not call('execute_transport', plan_id=plan['plan_id'])['executed']
    held = call('verify_grasp', duration_s=.1)
    assert not held['held']  # no acquisition/lift was actually commanded
    current = call('observe')
    placement = call('place_and_release', positions=[current['commanded_joints_rad']], durations_s=[.1],
                     open_joints_rad=q, release_duration_s=.1)
    assert not placement['released']  # still at initial support, not the placement
    assert not call('verify_placement')['placed']
    assert abs(call('observe')['time_s'] - start['time_s'] - .4) < 1e-8
    return {'checked': True}
""")
    write_json(
        candidate / "workflow.json",
        {
            "version": 3,
            "nodes": {
                "check": {"type": "script", "script": "policy.py", "inputs": {}},
                "done": {"type": "end", "status": "success"},
            },
            "edges": [["START", "check"], ["check", "done"]],
            "conditional_edges": {},
            "subgraphs": {},
        },
    )
    output = root / "skill-rollout"
    output.mkdir()
    result = rollout(root / "bundle", config, candidate, output)
    passed = (
        result["infrastructure_error"] is None
        and result["graph"]["graph_status"] == "success"
        and not result["verdict"]["task_completed"]
    )
    receipt = {
        "passed": passed,
        "graph": result["graph"],
        "infrastructure_error": result["infrastructure_error"],
        "task_completed": result["verdict"]["task_completed"],
        "scope": "All seven adapted operations through isolated GaP RPC; initial-posture negative control, not task success.",
        "external_author_calls": False,
        "workspace_checkouts_visible": False,
    }
    write_json(root / "skill-acceptance.json", receipt)
    print(json.dumps(receipt, indent=2))
    assert passed, receipt


if __name__ == "__main__":
    main(sys.argv[1])
