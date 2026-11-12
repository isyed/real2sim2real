# Controller and actor training

The [continuous object-reference route](../../docs/CONTINUOUS_OBJECT_REFERENCE.md)
places robot acquisition on the capture clock, followed by the demonstrated lift
and table return. It emits a planning-only artifact for fresh arm/finger admission
and evaluates complete free-object motion with acquisition-anchored retention.
The same route includes bounded unloaded-pad acquisition, optional non-pad finger
clearance fitting, and a saved-state position-motor audit. Geometry, stationary
retention, demonstrated motion and bilateral release have separate admission steps.
`closing_grasp_plan` adds joint closing-wrist and ten-finger proposals with optional
calibration from an actual loaded hold. Proposals remain coupled to bounded
object-frame wrist preload through arm-path admission and live finger ownership.
The [acquisition rollout search](../../docs/ACQUISITION_ROLLOUT_SEARCH.md) evaluates
bounded wrist/thumb/finger schedules in complete stationary physics, with identity
regression, full-hold contact scoring and unchanged task acceptance. Rejected
scores and path failures remain explicit; a new stationary pass still requires
moving/return/release validation.

The [CPU arm residual environment](../../docs/ARM_RESIDUAL_ADAPTER.md) adds a
reset/step interface for bounded bilateral pressure corrections on frozen arm
control. Its zero-policy validator compares full saved physics and unchanged
task outcomes. Observations are privileged; learner integration is separate.

The optional [pass-through GaP adapter](../../docs/PASSTHROUGH_POLICY_GRAPH.md)
executes frozen controller phases through one persistent zero-correction replay.
It verifies exact saved-state/record parity and zero added simulation time at
graph boundaries before candidate policy generation or skill adaptation.

ASL integration is an optional part of this module. The upstream author backend
remains separately installed; ordinary reconstruction and control do not import
it. The public integration components are:

| Component | Responsibility |
| --- | --- |
| `proposal_contract` | Standard-library validation of bounded parameters and immutable graph structure |
| `bounded_policy_graph` | Execute an admitted graph on the existing continuous simulator clock |
| `policy_feedback` | Export only named, pinned development runs with measured contact/retention diagnostics |
| `arm_motion_review` | Compare verified saved videos and original acceptance metrics |

See [bounded proposals](../../docs/BOUNDED_POLICY_PROPOSALS.md) and
[feedback-informed revisions](../../docs/POLICY_FEEDBACK.md). Author prompts,
credentials, dataset paths and experiment ledgers belong downstream. These tools
do not replace the simulator or perform residual RL training.

For a new grasp without a passing controller seed, see
[fresh grasp strategies](../../docs/FRESH_GRASP_STRATEGIES.md). This optional
interface compares deterministic and authored paired contact layouts through
one fitter and conditional fixture-hold evaluator, while retaining uncertain
surface evidence and rejected acquisition paths.

The [pipeline guide](../../docs/PIPELINES.md) separates human-reference training
from object-reference control. `object_control_stage` exposes arm-path screening
and actuator-driven evaluation through the common artifact/deadline protocol.
Reference extension preserves the admitted prefix; frozen-controller robustness
is evaluated separately from nominal success.

Optional [pad contact planning](../../docs/PAD_CONTACT_PLANNING.md) adds actual
collision-pad frames, normal fitting, closing-path checks and static wrench
diagnostics. Rejected hypotheses have their own static review; they are not
shown as successful physics rollouts.

For a CPU grasp/lift pilot driven by object geometry rather than a human motion
reference, see [object-centric grasp planning](../../docs/OBJECT_CENTRIC_GRASPS.md).
It provides a force-limited wrist/hand fixture, collision/approach checks, an
unassisted dynamic object, optional clearance feedback and parameter validation.
It is separate from the whole-body FlashCHORD/SONIC training path below.
For a second articulated hand supporting the lifted object's underside, see
[staged bimanual support](../../docs/BIMANUAL_SUPPORT.md). It preserves the primary
grasp and scene, checks insertion/parking collisions, measures actual upward
load sharing and validates frozen candidates against parked-hand comparisons.
Optional [common wrist coordination](../../docs/WRIST_COORDINATION.md) regulates
object pose using a shared bounded correction and joint-limit projection. Its
world-motion audit remains separate from contact and wrist-relative drift gates.
[Contact/posture search](../../docs/GRASP_POSTURE_SEARCH.md) evaluates finger
placement and reserved wrist travel against both retained controllers.
[Finger closing and contact feedback](../../docs/FINGER_CLOSURE.md) separates
unconstrained closing commands from actual loaded contact, with bounded named
offsets, per-finger surface regulation and numerical-integrity checks.
See the [task-first roadmap](../../docs/OBJECT_CENTRIC_ROADMAP.md) for optional
skill-graph integration and subsequent object-reference tracking.

`collision_cache_similarity.reuse` is an explicit offline alternative to a cold
collision decomposition when the new mesh is exactly the same triangulated
surface under a positive uniform scale and rigid frame change. It requires a
pinned source-cache hash, full corresponding vertices, identical valid faces
and a strict metric residual bound. It transforms the retained convex pieces
and records their new geometry/parameter key and hashes. Deformed meshes,
reflections and conflicting existing target caches fail. This preserves the
chosen proxy under the declared similarity; it does not claim a fresh CoACD
run would produce the identical partition. Use `collision_cache_policy.package`
to pin the result in a fresh portable policy, retain its source, and repeat
matching-device settling/physics. `rigid_collision_cache` remains strict about
rejecting scale changes.

`approach_experiment.ApproachExperiment` runs a bounded short-reference
balance/reach comparison against an unchanged retained5M actor. It prepares
an explicit native prefix, records the changed reference horizon, verifies
model/actor CPU forwards and builds fresh unassisted5M/conditional10M learners.
The host supplies guarded process execution and progress callbacks. Evaluation
uses three frame-zero trials, upright phase-matched contact and balance/reach
gates; only a new learner's matching state/replay can continue. It retains
the comparison actor and does not approve full-task deployment or submission.

`reference_window.prepare` produces an explicit native-frame diagnostic window
while preserving pose values and rebuilding the control clock. It copies only
declared assets, excluding stale derived parquets. `unassisted_pilot.configure`
sets both the object-assistance override and all curriculum scales to zero,
with explicit reset probability and bounded replay capacity. These helpers do
not establish full-task actor compatibility or training success.

Saved-state previews use guarded display decimation: a truncated object fails
the extent check, triggers a larger display budget, or falls back to source
geometry. Simulation and collision geometry remain unchanged. See
`display_geometry.json` in each new preview for the actual display choices.

Use `settling_compatibility.isolated_run_parameters` to match an isolated
settling probe to the controller's supported solver settings. The isolated scene
uses world-static supports because it has no reset loop to pin a kinematic body.
Record both solver parameters and backend with the result.

When retargeting changes only an object's body coordinate frame,
`rigid_collision_cache.reframe` preserves its validated collision hulls through
that rigid transform. Re-running convex decomposition can produce different
contact geometry even for the same visual surface. The helper requires matching
vertex correspondence/topology and rejects scaling, reflection and nonrigid
changes. Package its cache with `frozen.package(..., collision_cache=...)`, retain
the receipt, verify COM/inertia in the new frame, and repeat settling/physics
before training.

Keep runtime collision caches separate for each validated policy. A cache key
hashes mesh inputs and decomposition parameters; it does not uniquely identify
the resulting hull arrays. Separate runs can pin different hulls for the same
key. `policy_cache.seed_policy_caches` copies each policy's exact saved files
into `cache_root/<namespace>/convex`, checks all conflicts before copying, and
rejects an overwrite within a namespace. Set `FLASH_CHORD_CACHE_DIR` to that
policy's `cache_root/<namespace>` for both admission and training. Do not merge
all policies into one directory, choose a winner arbitrarily or recompute
validated hulls just to resolve a cache conflict. The helper is stdlib-only
and its receipt establishes byte preservation, not physical task success.

This module contains FlashCHORD simulation and the actual FlashSAC learner,
replay buffer, checkpoints and unassisted evaluation code.

The G1 recipe combines three components:

- **Frozen SONIC:** maps the body reference and robot state to body joint
  targets. It is a pretrained motion controller, not a task-language planner.
- **Fixed hand closing:** independently opens/closes left and right fingers
  from reference contact phase and actual palm proximity to the current
  object. Targets are rate-limited and bounded by joint limits.
- **Learned residual actor:** FlashSAC learns body and finger corrections on
  top of those targets. Zero residuals do not mean zero finger targets when
  the fixed hand controller is enabled.

FlashCHORD supplies parallel physics, observations and reward terms.
FlashSAC collects transitions into a bounded ring replay buffer, samples
minibatches, fits critics and updates the actor. Critic and replay state are
used for training; actor inference does not require the critic or replay.

The default generic `g1_bounded_flash_sac` recipe enables fixed hand closing,
uses 256 worlds and a one-million-transition replay capacity. It reduces
object assistance at 2M/4M transitions and removes it at 6M, with frame-zero
starts introduced early. These are experiment defaults, not established
success guarantees. Default rewards track pelvis/body/palms/object position,
encourage force closure/progress and penalize termination and rough actions.
Object orientation is an available objective option, not a hidden critic
input or automatically enabled additional reward.

Set an explicit `environment_steps` budget. The adapter rounds it down to a
multiple of the world count. `checkpoint_every` controls collection intervals.
Reaching a checkpoint does not select it for deployment: evaluate without
object assistance and compare multiple independent rollouts.

`scene.support` is false in the generic bounded recipe. Enable it explicitly
only with a verified robot-aligned `support.usda`. Tabletop tasks usually need
such a surface; the reconstruction module does not currently create it.

## Independent use

Use `examples/training.json` with a saved `robot-reference` artifact. Native
Hydra entry points remain available:

```bash
export R2S2R_ASSETS=/absolute/path/to/robot-assets
python -m real2sim2real.controller_actor_training.train --cfg job \
  experiment=g1_bounded_flash_sac
python -m real2sim2real.controller_actor_training.evaluate \
  evaluation.checkpoint=/absolute/path/to/policy_9999872.safetensors \
  evaluation.reference_artifact=/absolute/path/to/robot-reference \
  evaluation.world_count=1 logging.mode=disabled
```

The evaluation command launches physics and should use the same GPU
temperature/time supervision as training. Calling `--cfg job` only prints
configuration. Logging uploads are disabled by default.

For exact continuation, retain learner state and optionally replay via the
native `training.resume`, `training.checkpoint_path` and
`training.checkpoint.load_replay_path` settings. Saving replay is off by
default. Rebuilding a fresh buffer is possible, but changes the experiment.

When the task or reference horizon changes, `train_actor_warm_start` can copy
only a checkpoint's actor parameters and batch statistics into a fresh learner.
It verifies the checkpoint hash, exact action/observation schema and network
configuration. Critic, optimizers, replay, temperature and step counters remain
fresh. This is an explicitly declared transfer experiment, not exact resume.
The source checkpoint remains untouched and ancestry is saved in all new
checkpoints as `actor_warm_start_json`.

```bash
python -m real2sim2real.controller_actor_training.train_actor_warm_start \
  --actor /path/to/retained-policy.safetensors \
  --actor-sha256 CHECKPOINT_SHA256 \
  --transfer-evidence /path/to/transfer.json \
  --config-path /path/to/new-task-config --config-name train
```

The evidence JSON must declare `"task_changed": true` and should describe the
source/target task and reference horizons. Use a fresh training configuration
(`resume: false`), verify controller/geometry compatibility separately, and
evaluate the retained actor on the new task before updating it. Policy-based
replay warm-up and a conservative actor learning rate can preserve useful
behavior while a new critic learns, but neither guarantees preservation.

The GPU stack is pinned in `requirements-gpu.txt`. No pretrained weights,
episode-specific actors, datasets or cloud launchers are embedded here.

## Frozen baseline without actor training

Use the standalone `frozen`, `evaluate_frozen` and `render_frozen` entry points to
package SONIC plus fixed hands and evaluate deterministic zero corrections.
The portable bundle pins model/reference/robot assets and disables all assistance.
Saved kinematic targets and actual physical states are separate. See
[the frozen-controller workflow](../../docs/FROZEN_CONTROLLERS.md) for commands,
CPU video extras, supported scene scope and integration-test evidence.


Tabletop frozen baselines can run the standalone CPU `support_check` before GPU
evaluation. The matched video renderer includes the recorded triangle supports;
unsupported support geometry fails explicitly. Both helpers are scenario-neutral
and keep clean-target scoring and submission packaging downstream.

For an explicitly declared convex collision approximation, use
`collision_proxy.package` on a portable robot reference before frozen packaging.
It preserves visual geometry, motion and inertia, records its filled-concavity
assumption, and requires fresh physical validation. See
[placement tracking and first-grasp diagnostics](../../docs/PLACEMENT_TRACKING.md).

## Portable residual-actor pilots

Use `residual.configuration` to initialize fresh FlashSAC learning from a verified
frozen-controller bundle, and `evaluate_actor` to save unassisted physical states
for a matched before/after comparison. These helpers preserve the source duration
and fixed hand controller. See [the residual pilot workflow](../../docs/RESIDUAL_ACTOR_PILOTS.md)
for continuation, inference-only recording and CPU video commands.

For contact diagnosis, see [palm tracking and contact retention](../../docs/GRASP_DIAGNOSTICS.md).
[Bounded joint-reference experiments](../../docs/BOUNDED_JOINT_ABLATIONS.md) provide
explicit inference-only wrist/arm ablations, with unchanged checkpoint tensors
and separately recorded override provenance. These are opt-in diagnostics, not
default controller changes or guaranteed grasp improvements.

When changing object mass properties, use the
[retained-actor transfer workflow](../../docs/INERTIAL_ACTOR_TRANSFER.md) to
separate isolated settling improvements from controller performance. It includes
a strict inertial-only model audit and explicit deployment provenance.

## Task-specific contact admission

`contact_admission.assess(hand_contact_active, collision, expected_effectors=..., effector_audit=...)`
checks whether the configured controller can represent the interaction the task
requires. Hand phases are required for a hand task. A foot-push task can correctly
have zero hand phases: declare its foot effector, provide a source-geometry
proximity audit, and enable the corresponding robot–object collisions. The check
never invents contact labels or changes the controller. Downstream adapters must
bind the audit to the exact reference and policy hashes. Admission compatibility
does not certify scene stability, physical contact, task success or training
readiness; evaluate those separately.

`reward_alignment.audit(config, hand_contact_active, expected_effectors=...,
control_fps=...)` checks the separate question of reward/task alignment. Object
tracking can reward a foot push even without hand contact, while hand-only
force-closure and hand-contact counters do not measure foot contact. The audit
reports incidental hand phases, their enabled duration, missing collision scope,
and the absence of a dedicated nonhand-contact term. Admission and this audit do
not establish physical task success. Palm tracking follows the demonstration;
it is not a grasp requirement. The existing progress term measures episode time,
not independent object progress.

For a separately versioned **fresh** nonhand experiment,
`reward_alignment.without_hand_grasp_reward(config, expected_effectors=...)`
returns a copy with hand force closure disabled in the base objective and all
curriculum stages. It preserves body/palm/object rewards and reference phases.
It rejects hand tasks, disabled nonhand collisions, and learner/replay resumes.
This opt-in profile neither changes defaults nor adds a foot-contact reward;
retain a comparable first pilot before introducing additional shaping.

`planar_collision_base.propose(...)` is a bounded, experimental collision-only
hypothesis for an uneven reconstructed resting base. With a reviewed unit support
normal in object coordinates, it clips a declared band above the lowest point,
limits per-vertex correction and volume loss, and returns a separate closed
convex proxy. The caller must preserve visual geometry, poses and inertia in the
packaged model and verify those identities. A millimetre-scale placement gap does
not establish a stable contact patch or correct centre-of-mass support. Validate
the hypothesis with independent source evidence, placement and isolated CPU/GPU
settling; never adopt it automatically merely because it makes a box stand up.

`object_stability.assess_settling` reports independent limits on initial motion
and late drift, preserving the measured backend. A passing CPU screen requires
confirmation on the intended GPU backend. A failed isolated test remains useful
even when a robot rollout appears stable: the robot may be supporting the object.
Use the exact initial pose, rigid-object inertia, support mesh and collision
settings planned for training. Keep rejected geometry and the failed measurements
instead of widening limits to admit it. Settling admission alone does not certify
grasping, task success or actor-training readiness.

Isolated tests use world-static support geometry by default, avoiding free
support DOFs in a test without a reset/pinning loop. See
[static-support settling and initial-rest verification](../../docs/STATIC_SUPPORT_SETTLING.md)
for the standalone entry point, legacy diagnostic mode and required validation.

Flat mesh contacts can be tested with explicit `SimConfig.enable_multiccd=True`;
the legacy default is retained. `solver_ablation.package` prepares a separate,
audited frozen policy with only that contact-generation setting changed. Use
analytic controls and repeated isolated tests before new GPU/robot evaluation.
The setting does not change a reconstruction or automatically improve a grasp.

Flat objects can also rock because a mesh pair has only one contact point.
`SimConfig.enable_multiccd` and the isolated tester's `--enable-multiccd` expose
an explicit multiple-contact experiment; both default to false. Match the
controller's `impratio` when comparing isolated results. See
[controlled contact-mode diagnostics](../../docs/MULTICONTACT_SETTLING.md).
Use new artifacts and fresh physics results when changing contact mode.

`policy_settling` runs an isolated object test directly from a portable frozen
policy. It reads the actual initial pose, URDF/inertia, static support and solver
settings, including contact friction and free-object damping. For example:

```bash
python -m real2sim2real.controller_actor_training.policy_settling \
  --policy /path/to/policy --output /path/to/new-probe --device cpu
```

Use the intended GPU backend under the caller's thermal/time supervision for
confirmation. `POLICY_ADMISSION.json` binds the measured gate to the policy,
support and object model. CPU and GPU measurements remain separate; a failed
CPU screen is a warning requiring diagnosis and target-backend confirmation,
not permission to relax the limits. Pin the measured collision decomposition
when packaging so repeated probes and robot rollouts use the same hulls.

`policy_physics.match_scene(configuration, policy_config)` optionally checks
solver/timing parity and copies the policy's scene friction and free-object
damping into a separate training configuration. It refuses solver changes and
preserves world count, reward and curriculum settings. Its receipt lists actual
changed fields, which can be empty when the original recipe already matches.
This explicit check does not change default training recipes or certify task
success. Settling, complete frozen execution and successful manipulation require
separate evidence.

For bounded approach experiments, `approach_diagnostics.summarize` measures
time before a declared low-pelvis threshold, palm tracking error and sustained
hand/object contact during the corresponding hand's reference phase. Contact
after the first low-pelvis event is excluded from the useful-contact measure.
The threshold is a diagnostic, and contact alone does not certify grasping.
`approach_diagnostics.continuation` additionally checks the worst repeat so an
improved mean cannot conceal one severely failing trial. These opt-in analysis
helpers leave the learner, rewards, geometry and default recipes unchanged.

`continuation_curriculum.amend_reset_curriculum` provides an opt-in absolute-step
reset curriculum for an already unassisted learner. It preserves rewards and
reset-settling settings, updates both configuration views, and rejects invalid
or decreasing beginning-start schedules. Recovery starts can still have
reset-only assistance; this helper does not claim all training transitions are
unassisted. `continuation_review` distinguishes one noisy checkpoint from repeated
severe regression and supports a declared major review. These helpers do not
provision cloud resources, change default training, or certify task success.

`rollout_audit` analyzes immutable saved poses on CPU. Callers supply object-local
surface points, score thresholds and clocks. It reports full point error,
translation/rotation-only explanatory diagnostics, threshold accuracy, native
motion screens, evidence-derived phases and error intervals. It never changes a
trajectory or runs physics. Phase labels are inferred, motion warnings are not
acceptance gates, and counterfactual error decompositions are not physically
achievable policies. Dataset-specific scoring and candidate selection belong in
the downstream adapter.
# Long residual-training recipe

`real2sim2real.controller_actor_training.long_training.configuration` builds a
fresh or exactly resumed experiment from a portable tested policy. It preserves
that policy's scene, solver, contact settings and fixed-hand action. Object
assistance tapers to zero by10M collected transitions, including during recovery
resets; frame-zero reset probability is50% through40M and75% thereafter, with a
60M ceiling. `review` supports unassisted5M/10M diagnostic evaluations, a first
performance gate at15M and a major progress review at40M. These are experimental
defaults; task success and sufficient compute are not guaranteed. Episode
selection, budgets, cloud provisioning and competition scoring stay downstream.

## Bounded approach pilot

`bounded_approach.configuration` composes the portable policy's tested physics
with coarse pelvis/palm tracking rewards, fixed-capacity replay and a 15M
transition ceiling. Object assistance ends by 10M, reset assistance remains
zero, and 5M/10M are diagnostic checkpoints. The configuration preserves the
full reference and fixed-hand controller. Continuation requires the actual
matching checkpoint recipe, learner state and replay; changed buffer or optimizer
settings cannot silently reuse an old learner. `diagnostic_review` stops invalid
execution or severe combined score/balance regression. It does not launch work,
extend a run, certify task success or select a submission.

Contact-region audits and paired contact-arrangement diagnostics are documented
in [CONTACT_REGION_AUDIT.md](../../docs/CONTACT_REGION_AUDIT.md). These preserve
rejected plans and distinguish force balance from observed contact support and
validated two-hand robot execution.

Before fitting a handle hook or underbody support grasp, use
`handle_access_audit` to check opening geometry and swept tool clearance against
the source mesh, convex proxy and finite table. See
[HANDLE_INSERTION.md](../../docs/HANDLE_INSERTION.md) for the bounded probe
templates, browser review, uncertainty limits and integration with contact IK.

`guided_motion_pilot` compares a retained grasp against an opt-in reconstructed
object-motion transfer after support establishment. It preserves physical gates
and scores motion fidelity separately; it does not validate human finger motion.
See [GUIDED_MOTION.md](../../docs/GUIDED_MOTION.md) for initialization, observation
and Cartesian-fixture limits.

`grasp_retention` provides an opt-in, bounded primary-wrist correction for an
established loaded grasp. `retention_experiment` freezes the source trajectory
and physical criteria, then compares matched nominal and validation cases.
`retention_review` renders saved states with separate physical/fidelity outcomes.
See [GRASP_RETENTION.md](../../docs/GRASP_RETENTION.md) for privileged observations,
contact freezes, saturation and the distinction between retention and pad slip.

`retained_pad_search` reuses pad position/normal fits and closing-path checks
around a successful grasp before matched dynamics. `arm_path_audit` and
`arm_path_review` check an explicit local URDF against saved wrist paths,
including hand-frame compatibility, joint/speed limits and nonadjacent arm
collisions. See [ARM_PATH_AUDIT.md](../../docs/ARM_PATH_AUDIT.md). These are
kinematic integration diagnostics, not arm torque or physical execution claims.

For bounded support insertion and opposing-pad proposals against a retained
physical baseline, see [contact replanning](../../docs/CONTACT_REPLANNING.md).

[Contact-force feedback](../../docs/CONTACT_FORCE_FEEDBACK.md) adds bounded
finger-only pressure regulation, stationary-to-motion admission and saved-state
video comparison to the simultaneous bilateral fixture pipeline.
It also supports opt-in common object-pose corrections and planned-surface tangent
feedback. These are separately evaluated hypotheses; accurate object motion alone
does not establish retained contact or arm feasibility.

`grasp_sensitivity` performs frozen payload/friction tests from a pinned complete
passing fixture trial. It requires identical nominal replay, preserves reference
and physical criteria, and records each failed case before subsequent arm audits.

`payload_pressure` adds optional supported-load estimation and bounded pressure
allocation between hands. The core consumes force/acceleration observations;
the current fixture adapter changes only finger targets, records its privileged
inputs, and retains the original wrist path, actuator limits and acceptance gates.

`contact_patch_audit` adds read-only solved-contact material velocities and patch
centroids, preserving replay state archives. `bilateral_patch_replan` uses pinned
diagnostics to propose bounded pad-surface/finger-posture changes with fixed
wrists, then rechecks geometric admission before fresh physics. See
[loaded contact motion](../../docs/CONTACT_PATCH_MOTION.md).

`payload_posture` adds bounded supported-load-driven finger-posture transitions
between pinned admitted endpoints. The current adapter preserves wrist fixtures,
object targets and native limits. See [posture transitions](../../docs/PAYLOAD_POSTURE.md)
for configuration, observations and frozen-matrix admission.

`acquisition_compare` compares pinned, byte-identical contact replays at native
sample times. It separates acquisition/hold pressure, pad coverage, material
sliding, other-link forces and actual wrist-relative object poses, with an
interactive saved-data review. See [acquisition diagnostics](../../docs/ACQUISITION_DIAGNOSTICS.md).

`arm_finger_feedback` optionally binds the source bilateral pressure/load rule
to assembled arms. It reconstructs nominal finger waypoints before feedback,
preserves actuator ownership/native limits and labels live control separately
from archived command transfer. See [arm integration](../../docs/ARM_HAND_INTEGRATION.md).

`reference_extension` accepts an explicit longer `hold_s` to cover native-time
motion plus at least one second. It preserves the admitted reference prefix;
longer horizons are new evaluation cohorts, with unchanged physical gates.
An explicit `allow_prepend` can include earlier capture frames while verifying
that admitted capture poses/timestamps remain intact. It records a changed
reference origin and cannot claim the old control prefix or original capture-start
pose was reproduced. Robot acquisition remains an explicit stage.


Arm-mounted retention experiments can record read-only coherent pad motion with
`arm_replay_pilot.contact_patch_motion` and inspect acquisition/hold drift through
`arm_retention_audit`. Optional normal, contact-anchor and loaded-continuation
ablations retain force/joint limits and require fresh physical admission; they are
not automatically promoted controllers. See [arm integration](../../docs/ARM_HAND_INTEGRATION.md).
Use `stereo_patch_audit` to distinguish isolated supported surface points from
observed finite patch geometry before fitting hands. See
[contact-site search](../../docs/CONTACT_SITE_SEARCH.md).

`bearing_support_plan` provides bounded exterior top/underside support proposals,
real-pad fitting, insertion/seating collision witnesses and static load-path
screens. `bearing_support_review` produces a saved-data contact schematic and
rejection table. These stages never admit control or inherit a release pass;
see [bearing support planning](../../docs/BEARING_SUPPORT_PLANNING.md).
`paired_acquisition_audit --config <config.json>` checks a pinned retained primary
and replacement support moving together through pregrasp/open/closed states;
it preserves setup state and rechecks paired collisions and finite-patch balance.
`paired_contact_review --config <config.json>` projects those contact centres into
the matching saved calibrated RGB/depth and masks, retaining occlusion and missing
evidence explicitly. Both run under finite CPU deadlines and leave old results
unchanged. Neither claims loaded success or independently verified contact
patches. See [control readiness and preservation of earlier evidence](../../docs/CONTROL_READINESS.md).
`fixture_hold --config <config.json>` binds a paired hypothesis to a declared
staged or continuous-prefix stationary diagnostic with separately fixed physics
and feedback clocks. `fixture_hold_diagnostics --directory <trial>` separates
saved wrist tracking from grasp drift; `fixture_hold_review --directory <trial>`
renders CPU video without rerunning physics. New holds do not inherit earlier
arm, continuous-motion or release passes.
`arm_replay_review --refresh-page` updates protocol explanations using immutable
saved video, with prominent stationary hold/start/end labeling.
