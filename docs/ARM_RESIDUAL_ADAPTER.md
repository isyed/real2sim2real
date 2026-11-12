# CPU residual environment for arm-mounted articulated hands

This adapter accepts staged replay and complete
[continuous reference](CONTINUOUS_OBJECT_REFERENCE.md) motion with supported
finger release. Continuous stationary diagnostics remain excluded. Continuous
replays retain the original acquisition anchor, capture clock and release gate;
the added observations and rewards distinguish carrying from placement/release.

`ArmResidualEnv` supplies scalar NumPy `reset`, `step` and `close` methods around
the audited two-arm replay. It provides observations, bounded pressure actions,
provisional rewards and episode outcomes. The local UR/Sharpa embodiment comes
from the pinned replay; no robot assets are bundled. This module starts no
optimizer, GPU workload or hardware connection.

The environment shares the original replay loop through a coroutine. It pauses
before a controller update on each policy tick, receives a copied action, then
resumes the same controller and physics. Baseline configuration, reference,
duration, contact geometry, gains and task gates cannot be overridden through
the environment. Residual metadata is stored separately. A complete rejected
baseline is allowed for a bounded research pilot, with its rejection preserved.

## Actions, observations and clocks

Four normalized actions in `[-1, 1]` select additive pressure-target offsets:
primary thumb, primary fingers, secondary thumb, secondary fingers. Finger-group
pressure applies per pad, not as a total across four fingers. Positive increases
requested normal pressure; it does not directly prescribe a joint direction.

Declare `max_offset_n` (positive, <=2 N), `max_rate_n_s` (positive, <=10 N/s),
`active_phases` (close/lift/hold/return) and `policy_period_s` (10–100 ms, an integer
controller-period multiple). Offsets move toward the target at the controller
clock and return toward zero outside active phases. Nonzero target changes stay
between 0.1 and 10 N per pad. Actual measured force can differ from the target.

Release and settling cannot be enabled as pressure-action phases. Offsets decay
toward zero outside enabled phases, and the frozen supported-release controller
remains the sole owner of opening/unhooking. Once release starts it disables the
pressure servo and clears pad targets/offsets. This first continuous adapter lets
an actor improve acquisition/carrying; it does not learn a new release trajectory.

The existing live finger servo remains the sole finger actuator owner. Its native
force/joint limits, geometric guards and joint-offset/rate constraints remain.
There is no arm, object-state, reference-timing or simulator-parameter action.
Exact zero offsets skip the new transformation, preserving original rounding and
clamp behavior. Corrections are never added twice to archived corrected commands.

Observations include robot hinge positions/velocities, previous actuator commands,
object position/rotation errors with reference-validity flags, object positions
and drift in both wrist frames, pad normal forces and preceding targets, servo
offsets, load-sharing filter state, task phase/time and previous/applied actions.
The saved schema supplies vector slices and joint/actuator/pad ordering.

These are **privileged simulator observations**. A separate MuJoCo data object
computes coherent poses/contacts at the current physical state. No extra forward
solve modifies the live solver state. The observation is causal but not claimed
to expose every hidden controller state; a learner can use history. Hardware
sensing/noise/delay adapters are separate work. Prior commands are labelled as
prior commands, not expert labels or counterfactual current-state commands.

Staged drift anchors use the first recorded hold sample. Continuous anchors use
the first recorded sample at/after acquisition completion, matching the independent
task evaluator. Anchors never move when contact slips; a validity flag stays false
until the corresponding sample exists. Physics, controller and policy clocks remain
distinct.

Continuous observations add a six-stage one-hot vector (approach, acquisition,
carry, placement, release, settled), table contacts/load/clearance/object speed,
actual release latch/progress/support dwell/opening error, and total hand normal
force. The scheduled phase remains separately observable. A scheduled release
without the actual supported-release latch is still placement, not successful
release. The schema records all vector slices and stage ordering.

## API and evidence

```python
import json
import numpy as np
from real2sim2real.controller_actor_training.arm_residual_env import ArmResidualEnv

env = ArmResidualEnv(json.load(open("/data/residual-environment.json")))
try:
    observation, info = env.reset()
    while True:
        action = np.zeros(env.action_dim)
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
finally:
    env.close()
```

Required configuration: `baseline_result`, `baseline_sha256`, a new `output`
directory, `deadline_s` (<=900) and `residual` containing the four settings above.
The baseline result, scene and states are verified; replay also checks its audited
sources. Residual-on-residual baselines are rejected. Reset reconstructs fresh
physics, controller and filter state in a unique episode directory. An optional
integer seed is recorded, but this first environment has no randomization.
Different seeds must not be represented as independent robustness conditions.

Reaching the original horizon terminates, with separate `success` and original
`rejected_for` fields. Runtime deadlines truncate and mark the transition
ineligible for learning. Closing/resetting an unfinished episode writes an
interrupted receipt, not a passing result. Invalid actions are rejected before
advancing physics; completed episodes require an explicit reset.

The provisional reward is a time-normalized negative cost: object position
(weight 4, scale 20 mm), rotation (2, 10 degrees), both wrist-relative drifts
(6, 10 mm), missing opposing pad contacts (2), excess pad force above 10 N (1),
and normalized residual magnitude (0.1). Tracking/contact/drift costs apply in
hold. Rate bounds are enforced by the action filter; load sharing stays in the
baseline, without an additional reward term in this first adapter.

Dense learning costs are capped at 100 per complete horizon; raw acceptance
metrics remain uncapped. Complete success adds 10, complete rejection subtracts
10. Early physical failure subtracts 110, covering the remaining bounded cost
and failed-completion penalty. A higher reward never overrides a failed task gate.
For continuous tasks, tracking costs apply after approach, while grasp/contact
retention costs apply during carry, including descent. Clearance deficit applies
only on reference-defined suspended hold frames (weight 4, scale 10 mm). Placement
costs penalize missing table support and excessive clearance (weight 4). After the
actual release latch, hand force costs encourage unloading; settled opening error
is also penalized (weight 2). A table-penetration cost beyond 3 mm applies throughout
(weight 4). These provisional rewards do not replace the unchanged physical gates.

Outputs include the unchanged physical-state archive format, `result.json`,
`policy-trace.npz` with observations/actions, and `environment.json` with rewards.
Source code is snapshotted before execution. Incomplete traces remain diagnostic.

```bash
python -m real2sim2real.controller_actor_training.arm_residual_validate \
  --config /data/residual-environment.json
```

The bounded CPU validator runs the entire frozen horizon, compares every saved
state array with exact equality, and checks full diagnostic records, task verdict,
contact/arm metrics and physical scene. Any difference rejects reproduction.
A reproduction pass does not make the underlying task successful.

## Bounded pressure-authority trials

`arm_residual_trial` runs a declared deterministic correction policy over the
complete frozen horizon. It preserves the environment's original physical and
acceptance configuration and writes a separate `trial.json` with metrics,
protocol checks and policy provenance. A complete matched trial and the original
task verdict are both required for `acceptance_passed`. No optimizer runs.

```bash
python -m real2sim2real.controller_actor_training.arm_residual_trial \
  --config /data/pressure-trial.json
```

The trial configuration contains `environment` (the same environment settings
above) and `policy`. A constant policy is, for example,
`{"kind": "constant", "action": [0.5, 0.5, 0, 0]}`. With a 1 N environment
cap this requests +0.5 N on the primary thumb and each primary finger pad. The
environment still applies phase activation, slew bounds and native servo guards.

A causal diagnostic policy can use
`{"kind": "drift_ramp", "action": [0.5, 0.5, 0, 0],
"start_m": 0.004, "full_m": 0.008}`. It scales the requested correction from
zero to the declared action using the current primary wrist-frame drift norm.
It returns zero before the original fixed grasp anchor exists. Thresholds and
action directions must be declared before running the trial. Selecting them on
the same replay is controller development, not held-out robustness evaluation.

The trial snapshots its policy source before simulation, records action traces,
and checks original configuration, physical scene, controller/physics clocks and
the complete sampled timeline. Rejected or partial results remain visible.

Arm reviews and comparisons expose residual metadata. Existing controller-only
stationary promotion and robustness replay reject residual-policy artifacts,
because those runners cannot reconstruct an actor from the baseline configuration.
This prevents a successful learned policy from silently becoming a zero-actor run.

This is a scalar CPU environment with a Gym-style API, not a registered Gymnasium
environment or FlashSAC's vectorized JAX environment. Batched learner integration,
supervised demonstration import, domain randomization and hardware observations
remain separate work. G1/SONIC checkpoints are not compatible merely because
this adapter exists.

Continuous acquisition, demonstrated return and finger release require a separate
full zero-action verification for each frozen controller. The implemented
phase-aware extension preserves the staged path. See
[continuous tracking and residual-learning boundaries](CONTINUOUS_TRACKING_PROXY.md).
