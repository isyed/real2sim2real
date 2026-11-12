# Verify a graph without changing the controller

`passthrough_graph` and `passthrough_graph_validate` execute the actual optional
GaP v3 scheduler around one persistent arm replay. This first adapter admits only
a tool-only pass-through graph. It starts no author session, optimizer, policy
server, skill registry or RoboSimStudio process.

The physical controller remains `arm_replay_pilot`, accessed through the existing
`ArmResidualEnv` coroutine with every four-channel pressure action exactly zero.
That zero path returns no pressure modification. Robot assets, task geometry,
physical parameters, arm/finger servos, reference and acceptance limits come from
the pinned source. The only replay configuration change is its output directory.
Currently it accepts the continuous live-finger baselines supported by that
environment; arbitrary correction stacking is intentionally not supported.

## Shared clock across graph nodes

Each node calls `controller.advance_frozen_phase(phase)` once. Its phase comes
from the frozen source's ordered records, not a new set of durations. The first
node creates and resets one environment; all later nodes resume it. Both arms,
hand pressure filters, release latch and contact anchors remain in that same
session. The environment steps only when the tool resumes it with a zero action.

The node returns when the next scheduled phase is observed, without advancing
another step. The next node must start at that exact observation time and policy
step index. No settling operation or reset happens at a graph boundary. Phase
observations are quantized to the configured policy period; this does not delay
the original controller's faster internal transitions.

Graph execution takes wall time and is **not a real-time hardware scheduler**.
No simulation time passes while the graph transfers control between nodes. There
is no LLM call in the feedback loop. A scheduled release node is not evidence
that release physically happened: the original supported-release latch and
independent task evaluator still decide that.

The graph end node means execution completed. A rejected physical baseline must
still yield `candidate_success: false` even when graph execution and exact
reproduction pass.

## Contract and invocation

The config has exactly these fields:

```json
{
  "baseline_result": "/data/frozen/result.json",
  "baseline_sha256": "<SHA256>",
  "workflow": "/data/pass-through/workflow.json",
  "workflow_sha256": "<SHA256>",
  "output": "/data/fresh-run",
  "deadline_s": 900,
  "policy_period_s": 0.02
}
```

Build the graph with `workflow(phase_order(baseline["records"]))` from
`real2sim2real.controller_actor_training.passthrough_graph`, then pin the written
file's SHA256. It must exactly match this canonical phase chain. Extra waits,
scripts, tools, forks, resets, skipped phases and altered routing are rejected
before physics. This narrow contract is a reproduction gate, not the future
candidate-policy execution API or a general sandbox for generated Python.

```bash
python -m real2sim2real.controller_actor_training.passthrough_graph_validate \
  --config /data/pass-through.json
```

The entry point enforces a <=900-second process-tree deadline and CPU-only
environment. The original replay deadline also remains active. Interrupted,
truncated and out-of-order runs cannot pass. Existing outputs are never reused.

GaP and `graph-as-policy-core` are optional installed packages. The tested revision
is `09e45f8de27bc1a29629382ed7c2b9b4b384b3d6` from the dependency source pinned
by agentic-self-learning. Install the packages and their needed runtime imports
separately; do not upgrade the frozen MuJoCo/NumPy/SciPy environment to satisfy
the upstream full simulator extras. The adapter imports installed packages only;
it has no neighboring-repository import or symlink. Local robot skill checkouts
are not auto-discovered by this executor.

## Evidence

- `manifest.json`: frozen controller/protocol, graph, implementation snapshots,
  recursively verified source hashes, runtime versions and GaP source revision.
- `graph-trace/`: native GaP node execution trace.
- `replay/episode-0000/`: fresh simulated states, original diagnostics/verdict and
  the separate all-zero pressure-action trace.
- `verification.json`: exact equality of every saved qpos/qvel/ctrl/time value,
  compiled scene, original records, retention/reference/release diagnostics,
  controller updates and selected physical metrics; unchanged source hashes;
  one reset; contiguous graph boundaries with no extra policy steps.
- `interrupted.json`: retained failure when execution or reproduction fails.

Bit-for-bit agreement of saved values is a strict integration check, not proof
that every unsaved physics substep is identical or that the original grasp is
successful. The graph's zero actions do not establish useful correction authority,
robustness or hardware readiness. Candidate generation, skill adaptation and
nonzero residual control require separate experiments.

The next, separately bounded interface is described in
[Bounded policy proposals](BOUNDED_POLICY_PROPOSALS.md). It preserves this strict
pass-through path and admits only declared wrist/posture parameter changes.
