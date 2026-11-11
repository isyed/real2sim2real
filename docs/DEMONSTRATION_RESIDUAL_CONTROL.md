# Demonstrations followed by residual control learning

This is a proposed reusable experiment protocol, not an implemented UR/Sharpa
trainer or a validated reward configuration. Existing reconstruction, contact,
arm replay, diagnostics and FlashSAC components can be reused. An embodiment's
environment, observations, correction actions and learner integration still need
separate implementation and verification.

## What to learn

Keep the baseline controller, independent task reference and acceptance criteria
fixed. A small actor predicts corrections at a declared policy period:

`command = constraint_filter(baseline(observation, controller_state) + residual)`

The composition depends on the action: finger offsets are joint coordinates,
pressure changes are force targets, and wrist corrections are local SE(3) changes,
not additions to rotation matrices. Choose one command owner per quantity. Do not
apply corrections both to archived corrected commands and again to live feedback.
Do not give the actor authority over object state, reference timing, simulator
parameters or acceptance limits.

Begin with a small finger synergy/pressure action space that addresses the observed
failure. Retain wrist planning and existing object feedback. Introduce bounded
wrist corrections only as a separate ablation if finger actions cannot recover
the failure. Specify travel, rate, force, joint and collision bounds before
training; the sum of baseline and residual must satisfy them. Saturation and
intervention flags belong in the observations/logs. A zero residual must reproduce
the frozen controller exactly.

## Collect useful correction labels

A budget of 100–500 demonstrations can be a reasonable narrow-task pilot. It is
not a sample-complexity guarantee. Start with 20–50 instrumented episodes to check
observability, timing, retargeting and label quality, then collect more according
to held-out recovery performance. A thousand adjacent frames from one episode
are not a thousand independent scenarios.

Record successes, near slips, recoverable pose/load disturbances, interventions
and failures. Train behavior cloning on successful corrections and valid recovery
segments; retain failures for outcome modeling and evaluation rather than imitating
their actions as successes. Demonstrations that only execute an undisturbed task
teach little about recovery from the baseline's state distribution.

Robot teleoperation should log synchronized measured joint state, commanded
targets, estimated object/wrist poses, tactile or force observations, controller
internal state, task phase, reference, both clocks, action limits and interventions.
Record the baseline's counterfactual command on the same observations in shadow
mode, with its integrators/filters updated causally. For compatible additive action
coordinates, the supervised label is:

`residual_target = expert_command - frozen_command_on_the_same_state`

Use a transform logarithm in the declared local frame for pose actions. Actual
next joint position is not an expert command. Different gains, delays, frames or
action representations make naive subtraction invalid. If a desired expert action
lies outside the residual envelope, flag it as unsupported or redefine the study;
silent clipping would conceal that the chosen action space cannot imitate it.

Standard [UMI](https://umi-gripper.github.io/) records handheld parallel-jaw gripper
demonstrations without a robot. It can provide wrist/object strategy and motion,
but not directly the articulated finger/force corrections of a dexterous hand.
[DexUMI](https://dex-umi.github.io/) targets dexterous embodiments, but a new hand
still needs compatible sensing, retargeting and verification. Neither is a drop-in
source of robot residual labels. Prefer instrumented robot teleoperation for
fine contact corrections if the target hardware and sensing are available.

Real-robot teleoperation is already physical robot operation; this sequence can
precede autonomous policy tests, not all hardware use. If there is no validated
hardware interface yet, begin with simulation teleoperation or a bounded teacher
and explicitly treat its labels as simulation data.

Split entire episodes and collection sessions, not frames, into train/validation/
test partitions. Hold out physical conditions as well as random seeds. One possible
starting split is 70/15/15 percent, revised only before fitting; small test cohorts
must report uncertainty. Keep the final test cohort untouched by reward tuning.

## Actor observations and clocks

Use a short observation history: phase, baseline action, measured robot state,
estimated object pose/velocity error, wrist-relative drift, contact confidence,
estimated load sharing, tactile/force features, previous residual and saturation.
Include observation validity, timestamps and uncertainty. Exact simulator contact
forces, object poses and mass are privileged unless equivalent estimates exist
on the robot. Train with sensor errors, latency and dropouts; a privileged critic
may use simulator state while the deployed actor uses observable inputs.

Declare physics, low-level controller, sensor and actor periods independently.
A 20–50 Hz residual policy is a possible starting experiment over a faster
validated controller, not a universal rate recommendation. Smooth bounded outputs
between policy updates. Existing simulation frequency must not be mistaken for
a hardware command rate. Compare zero actor and learned actor at identical clocks.

## Proposed reward, separate from acceptance

Use dense costs for learning and an independent full-rollout evaluator for success.
A proposed initial cost rate is:

`L = 4 C_position + 2 C_rotation + 6 C_grasp + 2 C_contact + C_load + C_force`

`    + 0.1 C_residual + 0.1 C_residual_rate`

These are starting hyperparameters, not tuned or proven values. Define every term
dimensionlessly before changing a weight:

| Term | Proposed definition |
| --- | --- |
| Position | Squared object/reference distance divided by the declared position tolerance squared. |
| Rotation | Squared geodesic rotation error divided by the rotation tolerance squared. |
| Grasp | Sum of squared object drift in each wrist frame, relative to fixed admitted contact anchors, divided by each drift tolerance squared. |
| Contact | Missing/confidence-weighted opposing contact, active only when the task calls for load-bearing grasp. |
| Load | Deviation from the declared feasible two-hand sharing interval; do not demand equal force or total weight support while the table supports the object. |
| Force | Normalized excess over pad-force targets/limits; never reward force magnitude without an upper bound. |
| Residual | Mean squared action components normalized by their declared travel/force limits. |
| Residual rate | Mean squared change per second normalized by declared correction-rate limits. |

For a fixed task horizon H, use `r = -(dt_actor/H) * L`, plus a full-completion
bonus, for example +10, only when the independent evaluator passes the entire
declared trial. Reference time advances independently of the actor. Do not reward
an actor for freezing the task, reanchoring a slipping hand, stopping early or
shortening the terminal hold. Persist maxima/contact durations across the rollout;
recovering later does not erase an earlier peak-limit violation.

Use phase-aware clearance constraints: lifted transport requires table clearance,
while acquisition/placement permits intended table support. A lift task cannot
silently become a placement task with the same all-phase clearance reward.
Apply hard action constraints outside the reward. Unintended collision, lost
object support/drop, invalid simulation or actuator violations are explicit failed
trials; reward improvement cannot override that verdict.

If safety termination ends an episode early, avoid making it attractive by removing
future negative costs. Declare bounded per-step learning costs and choose a terminal
failure/absorbing-state cost at least as bad as the remaining bounded return, or
explicitly continue an absorbing failed state to the original horizon. Hard
evaluation retains raw uncapped errors even if a learning cost is capped. A generic
large negative constant with no horizon/rate analysis is insufficient.

## Learning and evaluation sequence

1. Validate the environment/action/observation adapter and exact zero-actor replay.
2. Fit bounded residual behavior cloning from synchronized successful corrections.
3. Evaluate frozen baseline and supervised residual on identical held-out cases.
4. Fine-tune that residual actor with simulator interaction (for example SAC),
   initially near demonstrated states. Keep the baseline frozen and use a declared
   decaying behavior-cloning anchor if needed; do not accidentally add a learned
   correction twice. Retain a separately frozen supervised checkpoint.
5. Randomize physically defensible mass/inertia, friction, starting pose, disturbances,
   sensing and delay. Match the intended hardware and vary combinations, not only
   one parameter at a time. Begin near nominal; expand only after measured recovery.
6. Compare zero actor, supervised residual and supervised-plus-RL residual using
   identical seeds/conditions, unchanged gates, complete horizons and multiple
   training seeds. Include held-out parameter combinations and sensing failures.
7. Review nominal regression, worst-case errors, success rates and uncertainty,
   then make a separate decision about supervised autonomous hardware trials.

Generalization means demonstrated recovery within this declared evaluation
distribution. It does not establish arbitrary new objects, tasks or environments.
The [residual RL formulation](https://arxiv.org/abs/1812.03201) and
[residual learning from demonstrations](https://arxiv.org/abs/2106.08050) support
the approach, but their results do not establish a required dataset size or
success for a new embodiment. A bounded research pilot can start from an imperfect
baseline; substantial training and deployment remain separate evidence decisions.
