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
geometry and exported object trajectories remain unchanged. The initial CPU
renderer supports mesh visuals and ground-only scenes; additional supports must
be explicitly rendered rather than silently omitted.

Application-specific episode identities, scoring-world alignment, submission
formats, independent clean-target scoring and online uploads belong downstream.
A completed rollout is not a successful manipulation or submission approval.
Three guarded 778-frame integration trials completed with zero residuals and
no assistance; both task-success measures stayed zero. Numerical and portability
checks passed nine tests, plus two CPU visual-transform tests. Both actual SONIC
models also passed offline CPU forwards. These establish the baseline's execution
and handoffs, not general grasp quality or hardware deployment readiness.
