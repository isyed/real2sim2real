# Residual actor experiments from a frozen controller

The controller-actor-training module already includes the FlashSAC learner,
critics, bounded replay, checkpoints and inference. `residual.configuration`
adds a portable handoff from a verified `frozen-controller`: it retains its
reference, duration-preserving control clock, robot/SONIC assets, fixed hand
settings and support choice, while starting a fresh actor/critic/optimizer/replay.
Explicit matching learner and replay paths enable exact continuation. Transition
budgets round down to whole parallel collections and the learning-rate schedule
stays fixed across the short continuations.

This optional recipe uses orientation guidance from the beginning (weight 0.1,
0.75 rad width), tight object-position guidance, 256 worlds and a one-million
transition ring buffer. Assistance decreases at 2M/4M and is zero after 6M.
These settings are experimental, not claims of successful grasping.

`evaluate_actor --policy FROZEN_BUNDLE --checkpoint ACTOR --output NEW_FOLDER`
records actual full-sequence frame-zero execution with assistance forced off.
It verifies the actor's reference/controller binding and saves physical poses,
contact counts and corrections separately from kinematic reset targets. It loads
the inference actor without the critic, optimizer or replay. `evaluate_frozen`
provides the matched zero-correction baseline. Run both with the same code and
portable bundle; a historical video with a different source/control clock is
not a matched baseline.

`render_frozen --policy BUNDLE --rollout ACTOR_ROLLOUT --baseline FROZEN_ROLLOUT
--output NEW_VIDEO_FOLDER` produces a CPU before/after video. It requires matching
body labels, reference hash and control-frame coverage. Rendering runs no physics
or training. The output keeps the historical `reference_vs_physics.mp4` filename;
its receipt labels the comparison as `frozen_vs_actor`.

Budget approval, cloud resource names, task-specific continuation gates, released
benchmark scoring and submission manifests belong to the downstream application.
Reward/diagnostic scores against a reconstructed reference are not official
competition scores. GPU execution still requires the caller's thermal/time guard.
