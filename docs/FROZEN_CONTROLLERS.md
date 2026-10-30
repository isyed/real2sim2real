# Frozen controller packaging and evaluation

The controller-actor-training module also supports a baseline with no correction
actor: pretrained SONIC body control plus fixed, rate-limited hand closing.
A deterministic zero correction vector leaves those controllers active. It does
not set finger targets to zero. Packaging and evaluation do not run training,
construct an actor/critic, or allocate a replay buffer.

Supply a verified `robot-reference` artifact and local G1/Dex3 assets. The
packager includes the reference closure, both SONIC models, robot URDFs and the
separate robot mesh directory. Every declared member is content-hashed. It
rejects missing/external URDF meshes and symlinks. Hard links are optional for
same-filesystem runs; deleting the original directory does not break the bundle.
Model/robot files are user-provisioned assets and remain outside ordinary Git.

```sh
python -m real2sim2real.controller_actor_training.frozen \
  --reference /path/to/robot-reference --assets /path/to/assets \
  --output /path/to/frozen-policy
# GPU evaluation: use the caller's temperature, clock/power and time guard.
python -m real2sim2real.controller_actor_training.evaluate_frozen \
  --policy /path/to/frozen-policy --output /path/to/new-rollout
# Optional CPU video extras: install this module with [video].
CUDA_VISIBLE_DEVICES='' python -m real2sim2real.controller_actor_training.render_frozen \
  --policy /path/to/frozen-policy --rollout /path/to/new-rollout \
  --output /path/to/new-video
```

The current recipe is G1/Dex3 only. Its configuration is composed directly from
the included Hydra recipe, never extracted from an episode-specific learned
checkpoint. The FlashSAC configuration group is selected explicitly so the
harness uses its injectable policy seam; the zero-policy loader creates no SAC
state. All object assistance, curriculum changes and actor resume paths are
disabled. Evaluation starts at frame zero in one world and records the full
reference. Source-frame playback follows the runtime's declared control clock;
inspect source/reference FPS and control FPS instead of assuming identical time
semantics across external datasets.

Outputs include `trajectory.parquet`, `metrics.json`, `audit.json`,
`physical_states.npz`, and separately `kinematic_states.npz`. Reference states
are produced by reset-to-target after the scored physical rollout. They are not
physical outcomes. The video uses one camera and display scale, with reference
on the left and physics on the right. Display meshes are simplified; simulator
geometry and exported object trajectories remain unchanged. The CPU renderer supports mesh visuals and self-contained triangle-mesh support
stages. Table colliders use the same transformed triangles as physics and the
recorded static support body; their hash is retained in the video receipt.
Unsupported primitive support types and external USD layers fail explicitly
rather than producing a misleading ground-only video.

Application-specific episode identities, scoring-world alignment, submission
formats, independent clean-target scoring and online uploads belong downstream.
A completed rollout is not a successful manipulation or submission approval.
Three guarded 778-frame integration trials completed with zero residuals and
no assistance; both task-success measures stayed zero. Numerical and portability
checks passed nine tests, plus two CPU visual-transform tests. Both actual SONIC
models also passed offline CPU forwards. These establish the baseline's execution
and handoffs, not general grasp quality or hardware deployment readiness.

## CPU preflight before scheduling physics

`controller_actor_training.verify_frozen.verify(policy, expected_frames=...)`
checks full G1/Dex3 reference coverage, zero reference fingers, every portable
URDF mesh path and actual finite forward passes through both SONIC ONNX models.
Set `CUDA_VISIBLE_DEVICES=''`; the function explicitly selects ONNX Runtime's
CPU provider. It returns model/reference hashes and geometric contact counts.
These checks establish loadability, not physical contact or task success.

Reusing a human conversion requires matching human parameters, model assets,
frame identities and world transform. A newer reconstruction with different human
parameters needs a new MHR-to-SOMA conversion even if the source video is the
same. Reusing an old physics rollout requires the exact same complete reference
and evaluator; a changed mesh or trajectory invalidates that comparison.


Table-support display has focused CPU tests for parent transforms, exact triangle
coordinates, missing recorded support bodies and external-layer rejection. These
checks establish display agreement, not true table accuracy or successful grasping.


## Initial placement on explicit support

Before a tabletop rollout, run the CPU-only placement diagnostic against the
portable robot reference:

```sh
CUDA_VISIBLE_DEVICES='' python -m real2sim2real.controller_actor_training.support_check \
  --reference /path/to/frozen-policy/reference/reference.parquet \
  --output /path/to/new-placement-report.json
```

This currently requires one rigid mesh-collision object and self-contained closed
convex triangle support components in the simulator's metre/Z-up frame. Object
samples include extrema and up to 2,048 indexed collision vertices. Surface
clearance and inside depth are measured separately, with explicit default limits
of 5 cm and 2 cm respectively. This samples prepared geometry; it neither proves
all triangle intersections absent nor certifies that the reconstructed table is
correct. Invalid/open/nonconvex support solids fail rather than becoming a
misleading convex penetration score. USD inspection uses the simulator's runtime
USD dependency; no GPU execution is needed.

## Different camera and controller frame rates

Packaging compares the native reference rate with SONIC's 50 Hz control rate.
For a different rate, it retains the native reference and prepares a separate
control-rate reference, preserving clip duration and endpoints. Joint positions
are interpolated; orientations use SLERP; contact labels use nearest timestamps.
Palm targets are recomputed from interpolated joints by CPU forward kinematics,
then checked with the normal strict palm validation. Interpolating palm poses
independently can violate that check even when both source poses were correct.
Optional per-hand fields containing only nulls are treated as absent. Populated
unsupported hand-geometry fields fail explicitly rather than retaining a stale
frame count.

The evaluator retains `control_trajectory.parquet` and physical states at the
controller rate, then exports `trajectory.parquet` at native source timestamps.
Positions interpolate between saved achieved poses and rotations use SLERP;
source reference poses remain separate. `source_clock.json` records both clocks
and both hashes. This mapping preserves physical tracking errors; it does not
replace achieved motion with reference motion. Videos use the control-rate states
and the actual simulation duration. Old packages that replay unequal source and
controller frame rates one-for-one are rejected and must be repackaged.
