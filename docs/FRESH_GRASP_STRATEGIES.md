# Fresh grasp strategies on a fixed reconstructed scene

`controller_actor_training.grasp_strategy` evaluates a new bilateral contact
proposal without requiring a successful parent controller. It complements the
[baseline-seeded graph adapter](BOUNDED_POLICY_PROPOSALS.md): the author selects
contact layouts and wrist/opening parameters instead of adding small corrections
to an existing robot plan. It does not learn a neural controller or replace the
simulator, robot model, IK solver or task evaluator.

## Scope and uncertainty

The current strategy family is simultaneous two-hand local-wall pinching, with
thumb/index/middle collision pads on each hand. Both hands may select distinct
geometric layouts, thumb sides, wrist rolls, opening gaps, preload, flexion seeds
and above/outward acquisition. Layouts come from source-mesh local solid spans;
they are not previous fitted grasps. Arbitrary contact coordinates, object tilting,
new pressure laws, controller timing changes and script generation are excluded.
This is a bounded fresh strategy, not unrestricted policy discovery.

The source scene and settled initialization are hash pinned. The explicit
`multi_hand_scene.append_hypothesis_hand` entry point composes the second hand
without fabricating a successful parent. The existing `append_hand` retains its
passing-parent requirement. Composition verifies that original named-body,
geometry, joint, actuator and solver properties are preserved. Force-limited
Cartesian wrist fixtures do not establish arm reachability.

Observation admission remains separate from mechanics. A reconstructed surface
sample can support a numerical contact hypothesis without proving actual wall
thickness, finite bearing area or initial robot access. Every new result keeps
`control_ready=false`; even a simulated hold pass cannot validate hidden geometry.
The gravity-settled condition is explicitly different from reference-preserving
initialization. Full demonstrated motion needs separate admission.

## Interfaces

| Component | Purpose |
| --- | --- |
| `grasp_strategy_contract` | Standard-library validation of the complete two-tool graph and bounded inputs |
| `grasp_strategy catalogue` | Freeze a bounded local solid-span catalogue without prior controller outcomes |
| `grasp_strategy.deterministic_proposals` | Rank geometry-only comparison proposals without feedback |
| `grasp_strategy evaluate` | Execute native fitting and conditional hold through the actual GaP scheduler |
| `grasp_strategy_feedback` | Export only named, pinned development results; retain missing physics as missing |
| `grasp_strategy_review` | Compare static fits and, only when executed, saved-state hold videos |

The graph is `START → plan → hold → complete`. `grasp.plan_pair(hands)` fits one
immutable pair. `grasp.test_hold()` either records why physics was skipped or runs
one complete stationary test. One successful graph traversal is not task success.
The author cannot change tools, branching, physics, evaluation or execution code.

The same fitter handles deterministic and authored proposals. It uses native
joint limits and pad normals, checks the entire sampled paired acquisition path,
and combines actual model-pad surface samples under shared per-pad force budgets.
Missing pad samples do not establish that a complete grasp is physically
impossible; they reject the candidate before the balance calculation can certify
it. Missing geometry or excessive collisions prevent the hold from starting.

## Running a downstream campaign

Provision a prepared object scene, local robot assets and the optional MuJoCo/GaP
runtime. These are not bundled or downloaded by the module. Private dataset paths,
author prompts, credentials, archived model calls and campaign budgets belong in
the downstream repository. The public package never imports that repository or
the author checkout.

```bash
python -m real2sim2real.controller_actor_training.grasp_strategy prepare --config /data/scene-config.json
python -m real2sim2real.controller_actor_training.grasp_strategy catalogue --config /data/catalogue-config.json
python -m real2sim2real.controller_actor_training.grasp_strategy evaluate --config /data/proposal-config.json
```

An evaluation config supplies `catalogue: {path, sha256}`,
`workflow: {path, sha256}`, a fresh `output` and `deadline_s <= 900`.
The catalogue pins the scene configuration, receipt and initialization.
The CLI applies a CPU subprocess deadline. Direct API callers must supply their
own finite process bound. Feedback and review exports are read-only analyses and
also require downstream deadlines.

The scene config fixes force/torque limits, material assumptions, IK budget,
sampled collision limits, staged acquisition/lift/hold clocks and simulation step.
The author cannot alter any of them. Historical lift-only tasks without a
`goal.hold_s` use the explicitly declared positive rollout hold duration; when
the task supplies a minimum, the rollout cannot shorten it.

If geometry passes, the fixture trial preserves native contact, drift, rotation,
support and penetration checks. The strategy wrapper additionally requires the
complete requested duration and at most 10 mm wrist-relative drift measured from
acquisition completion through the hold, separately for both hands. No re-anchoring
at the first hold sample may hide early slip.

## Feedback and selection

Start from an empty author workflow for the fresh condition. A revision receives
only declared development results and a prior graph. Record every proposal,
rejection, model call and infrastructure failure. Never count a skipped hold as
a simulated failure rate, or a passed data contract as a fitted grasp.

The feedback distinguishes per-hand target errors and missing model-pad samples
from paired collisions and gravity balance. Unexecuted physics contributes no
measured load, slip or success observations. The comparison requires identical
scene/configuration/catalogue pins and preserves all original verdicts.

Feedback and review configurations must pin both `evaluation: {path, sha256}`
and its sibling `plan: {path, sha256}` for every named run. The shared loader
rejects changed inputs, mismatched directories and contradictory verdicts before
export; it rechecks those pins after export. Pinning the evaluation alone does
not protect the fitted poses or collision diagnostics stored in its plan.

A deterministic comparison is optional; it is not a grasp-admission gate. Equal
candidate counts help distinguish adaptive proposal gains from merely trying
more candidates, but do not equalize model inference, fitting effort or wall
time. Freeze the ranking rule before examining results and record any later
budget extension explicitly. Expanding the count must retain the existing
proposal prefix. Keep model usage and runtime separate from candidate counts;
one adaptive chain on one scene is development evidence, not a general success
rate. Never send comparison outcomes to a feedback condition that excludes them.

The current deterministic proposer ranks pairs at least 35 mm apart by
`distance(pair_midpoint, assumed_COM) - 0.1 * pair_separation`. It uses fixed
thumb sides, zero wrist roll, 8 mm opening, 1 mm preload and a 0.7 rad flexion
seed. The author can choose these parameters within the contract and revise them
after measured failures. Both outputs are inputs to the same numerical hand
fitter; neither directly supplies a learned joint controller. ASL selects bounded
parameters here, rather than generating arbitrary manipulation code.

This comparison does not represent every conventional planner: a numerical
optimizer or adaptive search could also vary wrist orientation, finger seeds
and contact layouts using prior evaluations. A fixed-ranking comparison tests a
specific inexpensive heuristic. Report it as such, rather than interpreting a
better authored proposal as superiority over deterministic robotics in general.

Set the feedback budget and stopping rule before authoring. Retain the strongest
evaluated partial result rather than automatically replacing it with the latest
revision. Regressions, uncertainty and held-out checks remain visible. Advance
to real arm paths and loaded arm holds only after the relevant geometric and
fixture results warrant those separate tests; see [control readiness](CONTROL_READINESS.md).
