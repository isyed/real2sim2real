# GaP workflow format (v3): the policy you are writing

A policy is a **directory**: `workflow.json` + `scripts/*.py`. The graph runs in
the GaP runtime against a simulated robot. Its nodes either call **tools** (the
complete list is in `TOOLS.md`) or run your **scripts**.

## workflow.json

```json
{
  "version": 3,
  "meta": {"name": "...", "description": "..."},
  "nodes": {
    "locate": {"type": "subgraph", "ref": "locate_sg"},
    "done":   {"type": "end", "status": "success"},
    "abort":  {"type": "end", "status": "failure",
               "recovery": [{"tool": "robot.open_gripper", "inputs": {}}]}
  },
  "edges": [["START", "locate"]],
  "conditional_edges": {
    "locate": {"router_field": "exit",
               "mapping": {"ok": "done", "failed": "abort"}}
  },
  "subgraphs": {
    "locate_sg": {
      "skill": "generated",
      "inputs": {},
      "outputs": {"box": {"$ref": "fit.box"}},
      "nodes": {
        "read": {"type": "tool", "tool": "sim.get_object_obb",
                 "inputs": {"object_name": "cube"}},
        "fit":  {"type": "script", "script": "scripts/locate/fit.py",
                 "inputs": {"obb": {"$ref": "read.obb"}}},
        "ok":   {"type": "noop"}
      },
      "edges": [["START", "read"], ["read", "fit"], ["fit", "ok"], ["ok", "END"]],
      "conditional_edges": {},
      "exit": {"router_field": null, "success_values": ["ok"]},
      "on_error": "failed"
    }
  }
}
```

Rules that matter:

- **Dataflow is `$ref`.** `{"$ref": "node.field"}` reads another node's output
  field; `{"$ref": "in.name"}` reads a subgraph input. Dotted paths index into
  results (`plan.trajectory.waypoints.0`). Literals are plain JSON.
- **Top-level nodes are subgraphs and ends.** Each subgraph declares `inputs`
  (typed names, bound at the top level from an earlier subgraph's `outputs`),
  `outputs` (`$ref` into its own nodes), `exit` (`success_values`: which
  terminal node names mean success; each must exist as a node, usually a
  `noop`), and `on_error` (the exit label taken when any node inside raises).
  A top-level subgraph node binds its inputs like this:

  ```json
  "grasp": {"type": "subgraph", "ref": "grasp_sg",
            "inputs": {"box": {"$ref": "locate.box"}}}
  ```

- **Subgraph input types are a fixed vocabulary**, checked by name. Scalars:
  `str`, `float`, `int`, `bool`. Structures: `Pose`, `Vec3`, `Quaternion`,
  `OrientedBoundingBox`, `Trajectory`, `GraspCandidates`. There is no `dict`,
  `Any` or bare `list`; a pose-shaped value is a `Pose`.
- **Routing.** The top-level `conditional_edges` map a subgraph's exit label to
  the next node, with `"router_field": "exit"`. Every exit label
  (`success_values` and `on_error`) needs a mapping.
- **Failure surfaces only by raising.** A subgraph's `on_error` label is a
  symbol, not a node: no node may have that name and nothing inside may route
  to it. To take the failure exit, raise an exception in a node script.
- **Loops are backward edges** at the top level (map an exit label back to an
  earlier node).

## Node scripts

```python
from typing import TypedDict

from real2sim2real.agentic_policy._vendor.gap import NodeContext


class Output(TypedDict):
    hover: dict


def run(ctx: NodeContext, obb: dict, height: float = 0.10) -> Output:
    c = obb["center"]
    frame = ctx.tool("robot.grasp_frame")  # the hand pointing straight down
    hover = {"position": {"x": c["x"], "y": c["y"], "z": c["z"] + height}, "rotation": frame["rotation"]}
    plan = ctx.tool("motion.plan_joint", pose=hover)
    if not plan.get("trajectory"):
        raise RuntimeError(f"no plan to the hover: {plan.get('reason')}")
    ctx.tool("robot.execute_trajectory", trajectory=plan["trajectory"])
    return {"hover": hover}
```

- The signature is `run(ctx, **inputs) -> TypedDict`. Declare an `Output`
  TypedDict; its keys are what `$ref` can read from this node.
- Call only tools listed in `TOOLS.md`, always with a **literal** name string.
- Raise to take the subgraph's `on_error` exit; return normally to continue.
- Poses are `{"position": {"x","y","z"}, "rotation": {"w","x","y","z"}}` in
  the world frame, metres.

## What success means

The task's evaluator scores every episode, not your graph. Reaching an `end`
node with `status: success` is a claim; only the simulator's verdict counts.
Motion tools block until the arm settles, so one move is tens to hundreds of
control steps, and every episode has a step budget.
