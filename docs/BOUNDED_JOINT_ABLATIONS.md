# Bounded joint-reference inference experiments

`controller-actor-training` supports an opt-in correction toward existing joint
reference angles. The default gain is zero. An explicit joint-name subset can
isolate a wrist or arm while leaving other joint commands and action-history
entries untouched. This does not apply object forces or teleport the robot.

For actor evaluation, pass `--controller-override override.json` to
`real2sim2real.controller_actor_training.evaluate_actor`. The file must contain
exactly `reference_pull_gain`, `reference_pull_max_rad` and
`reference_pull_joint_names`. Gain is in [0, 1]; the positive angular bound is
at most 0.3 radians. The subset must be nonempty, unique and valid for the robot.
Checkpoint tensors, reference and scene remain unchanged. The evaluator checks
the original checkpoint against the portable policy before applying the override
and records the override hash and values separately in `audit.json`.

The correction is `clip(gain * (reference_target - joint_target), ±bound)`.
It is a per-step target adjustment, not a guarantee about actual wrist angles,
contact forces or grasp success. Reference targets are not automatically verified
grasp locations. Compare unchanged-controller trials using the same evaluator,
inspect object-relative palm errors and physical contact intervals, and verify
balance and task tracking before retaining a variant. Repeat apparent gains;
retain rejected trials without replacing the input policy.

Saved-state comparison videos derive labels from both rollout audits so an actor
baseline is not incorrectly labelled as a frozen-only controller.
