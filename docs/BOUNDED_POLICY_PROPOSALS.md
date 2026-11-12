# Bounded, baseline-seeded policy proposals

The first author experiment changes only the parameters of an existing controller.
It compares a frozen controller with one generated proposal in the same scene,
physics runtime, robot embodiment, native reference timing and task evaluator.
This is an ASL-derived seeded proposal condition, not a reproduction of the
upstream RoboSimStudio benchmark or an RL-trained policy.

`proposal_contract` supplies a standard-library-only validator that can also run
inside a clean author workspace. `seeded_workflow` adds one configuration node to
the canonical pass-through graph. An author may change only that node's inputs:

- Object-frame wrist translation vectors, norm <=6 mm per wrist.
- Object-frame rotation vectors about each wrist origin, norm <=0.06 rad.
- Explicitly admitted finger flexion offsets, <=0.06 rad per joint and
  <=0.12 rad per hand.

Graph structure, phase names, tools and all other inputs are immutable. Scripts,
extra waits, loops, resets, controller gains, physics and acceptance changes are
rejected. The proposal is data; the executor never imports author-written code.
The existing [bounded grasp adapter](BOUNDED_GRASP_CORRECTIONS.md) additionally
checks native joint bounds, correction speed, wrist IK and sampled command-path
collisions before executing the original free-object dynamics.

The offsets rise during the original closure and recover to zero before release.
The existing pressure servo remains the sole finger owner. Table-support gates,
thumb unhooking and withdrawal keep their original commands. These are fixed
posture offsets, not a newly trained feedback controller.

## Continuous graph execution

`bounded_policy_graph.NativePhaseSession` drives `arm_replay_pilot.rollout` with
an optional `phase_handoff` interface. It yields only on native phase changes and
accepts no action when resumed. There is one simulator initialization and one
persistent controller state. Graph handoffs add wall-clock overhead but no
simulation steps or pauses. This interface cannot be combined with the pressure
residual action interface.

Before using a proposal, run an all-zero graph and compare every saved qpos,
qvel, control and timestamp, the compiled scene, all original diagnostic records,
release metrics and physical verdict with the pinned source. Graph completion
must remain distinct from task acceptance. The graph does not provide a hardware
real-time scheduler.

## Invocation

```json
{
  "baseline": {"path": "/data/frozen/result.json", "sha256": "..."},
  "workflow": {"path": "/data/proposal/workflow.json", "sha256": "..."},
  "allowed_joints": ["right_index_PIP", "support__left_thumb_IP"],
  "output": "/data/fresh-candidate",
  "deadline_s": 900
}
```

```bash
python -m real2sim2real.controller_actor_training.bounded_policy_graph \
  --config /data/proposal-run.json
```

The runner uses the installed optional GaP scheduler with one worker and an
explicit two-tool registry. No skill registry is discovered. It verifies the
pinned recursive input chain, preserves the baseline configuration and rejects
existing output directories. The entry point bounds CPU execution to at most
900 seconds. The existing simulator's early-termination and acceptance checks
remain active. A failed path or incomplete rollout is a rejected proposal, not
permission to silently generate another one.

Graph completion also rejects a native `incomplete_pilot` verdict, including an
abort within the final release-settling phase. Phase timestamps must be finite
and strictly advance. The in-process `run()` function validates the same config,
but its caller must supply the process deadline; prefer the bounded CLI.

`manifest.json`, the graph trace, ordinary replay artifacts and `receipt.json`
record the proposed parameters, provenance, phase boundaries and original task
verdict. `interrupted.json` retains integration or admission failures. Every
candidate has `promotion_eligible=false`; a relative improvement still needs
paired complete holds, full motion and robustness checks before promotion.

## Author and evaluation separation

Stage a fresh author packet containing only the seed, tool contract, planned
scene geometry and declared limits. Keep prior simulations, failure diagnoses,
held-out cases and evaluator internals outside the author's filesystem. Archive
that packet and its hashes before calling the author. One zero-feedback
experiment means one final proposal and no physical-result-driven repair; a
syntax check is not a simulation-feedback iteration.

A downstream campaign may invoke upstream ASL's author backend, but must preserve
its actual provenance and record modifications to the harness. Upstream's default
nonempty-seed wording says the checker refused the policy; correct that wording
for an expert baseline seed. Do not infer a successful author call solely from
ASL's returned object: inspect transport errors, token usage and the final graph.
A failed network launch with zero model tokens is infrastructure failure, not a
policy outcome. Keep it in the ledger.

Compare admitted complete rollouts with `arm_motion_review`. It requires equal
native clocks and verified rendered videos. Its optional `description` describes
the particular experiment and is HTML escaped. Do not label a zero bridge replay
as an agent-generated policy, or claim a general method winner from one proposal.

For a portable comparison, set `combined_video: true` in `arm_motion_review`.
It exports a labelled side-by-side MP4 and verifies the decoded frame count.
Different input sizes or frame rates are rejected; no frame or simulation time
resampling occurs. The HTML also reports observed drift, clearance, reference
errors, penetration and release rather than relying on video appearance alone.
The reusable `real2sim2real.review_comparison.side_by_side` helper is independent
of the simulator. It verifies both inputs remain unchanged, rejects unequal frame
counts, publishes only after successful decoding, and never overwrites an input
or an existing output. Failed exports leave no partial comparison at that path.

When isolating an upstream author CLI, test read access, permitted write access
and hidden-workspace boundaries without a model call first. Bundle required CLI
companion binaries, and pre-create read-only protected mount points if the inner
sandbox needs them. Record each infrastructure attempt separately from the count
of generated and evaluated proposals; setup retries can consume model tokens even
when no policy is produced. Never send simulation outcomes to a fresh proposal
condition unless that feedback is explicitly part of the declared experiment.

For a separately declared feedback condition, use the
[measured policy-feedback exporter](POLICY_FEEDBACK.md). It retains the original
acquisition anchor, measured contact semantics and native source verdicts.
