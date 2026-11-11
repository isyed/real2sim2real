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


## Residual reinforcement learning and generalization

An actor added to a frozen controller is residual reinforcement learning when it
learns its correction from rewards obtained by interacting with the environment:
`action = frozen_action + bounded_residual(observation)`. A network trained only
to imitate stored corrections is supervised learning instead. Conventional
control plus learned residuals is the formulation in
[Johannink et al., Residual Reinforcement Learning for Robot Control](https://arxiv.org/abs/1812.03201).
The method does not mathematically require a perfect baseline; a project may
require nominal acceptance before substantial training to isolate reconstruction,
contact geometry or actuator-model failures first.

Generalization requires an explicit distribution of payloads, friction, pose
errors, disturbances, sensor errors and delays during training, followed by held-out
conditions and random seeds. A single nominal run does not establish robustness.
Robustness within tested variations does not imply handling arbitrary new objects
or tasks. Keep residual rate/travel bounds and actuator constraints in the action
interface, keep acceptance criteria outside the reward, and evaluate the frozen
controller with zero residual on the same cohorts.

Simulator contact forces, exact object poses and acceleration may be privileged
observations. A hardware policy needs equivalent estimated/sensed inputs and
delay/noise tests. Existing FlashSAC infrastructure does not by itself supply a
UR/Sharpa training environment or a compatible action/observation interface;
those adapters need implementation and validation before checkpoint reuse.

The [CPU arm residual adapter](ARM_RESIDUAL_ADAPTER.md) now supplies a bounded
pressure-action reset/step environment on the shared arm-replay loop, with explicit
privileged observations and exact zero-policy comparison. It does not connect the
scalar CPU environment to the vectorized G1/FlashSAC learner or start training.

For a proposed supervised warm start followed by residual RL, see
[demonstration and reward design](DEMONSTRATION_RESIDUAL_CONTROL.md). That protocol
defines correction labels, data splits, observable inputs and phase-aware rewards;
it does not claim that a new embodiment's training adapter is implemented.
