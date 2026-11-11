# Acquisition search in complete simulated rollouts

`controller_actor_training.acquisition_rollout_search` ranks bounded robot
acquisition schedules using actual unassisted stationary arm dynamics. It never
changes the original task verdict, raw object-reference clock or retention anchor.
It complements the geometric closing-grasp planner: a static fit seeds the search,
but every scored candidate must complete acquisition, early lift and the entire
original stationary hold.

## Robot schedule contract

Arm replay accepts optional `acquisition_profile`:

```json
{
  "wrist_windows": {"right": [0.0, 0.75], "left": [0.0, 0.75]},
  "finger_windows": {
    "primary": {"thumb": [0.0, 0.5], "fingers": [0.35, 1.0]},
    "secondary": {"thumb": [0.0, 0.5], "fingers": [0.35, 1.0]}
  },
  "pressure_activation_fraction": 0.5
}
```

Windows are fractions of the existing close interval, not seconds or reference
timestamps. Starts are 0–0.5, ends 0.5–1, and widths at least 0.4. Open and final
closed postures are preserved. Optional `joint_windows` maps explicit owned
finger-joint names to individual windows, permitting proximal/distal sequencing
without naming a particular robot in the reusable module. Unknown, arm or object
joint overrides are rejected. `pressure_activation_fraction` optionally selects
50–100% of closure for the original live pressure rule; it cannot alter force
goals, gains, normal-speed, joint-correction or actuator budgets. The resulting
force settings must still satisfy their original validator.

`acquisition_profile.identity()` preserves the original arm/finger commands exactly.
Binding occurs once on a fresh live controller; it cannot share closure timing
with a separate `finger_close_fraction` owner. Pressure corrections remain with
the live finger controller. Finger window interpolation uses the original
open/closed endpoints and returns to the ordinary hold/release rule afterward.

Only robot arm commands within the original acquisition interval are resampled.
The audited joint path supplies their poses; raw object-reference timestamps,
phase labels, frame IDs and score cohorts remain unchanged. New native arm speed
and joint-margin checks are followed by dense full arm/finger/environment
collision screening. This reuses source IK poses and screens their new timing;
it does not claim a new Cartesian IK solve. Nominal finger/arm self-collision
remains grounds for preflight rejection. Dynamic collisions and all original
task gates still apply throughout physics.

Every new profile requires complete matching stationary admission before moving
replay. A stationary pass does not validate full demonstrated motion, table return,
release, payload/friction robustness or hardware execution.

## Finite search contract

Required search config fields:

| Field | Meaning |
| --- | --- |
| `seed_config` | Pinned `{path, sha256}` of an existing live-finger stationary config |
| `baseline_result` | Pinned complete physics result from that same controller |
| `identity_result` | Pinned identity-profile replay with exactly matching saved states |
| `profiles` | 1–16 distinct `{name, settings}` candidates |
| `parallel_trials` | 1–3 bounded local CPU trials |
| `deadline_s` | Overall finite budget, <=7,200 s |
| `trial_deadline_s` | Per-child budget, <=900 s, shortened by remaining overall time |
| `output` | New artifact directory |

Optional pinned `parent_search` records adaptive development provenance.

Candidate names cannot reuse protocol/progress/search artifact names or collide
with another candidate's automatic `-repeat` output. These checks occur before
the search output is created.

Invoke:

```bash
python -m real2sim2real.controller_actor_training.acquisition_rollout_search \
  --config /private/path/search.json
```

The runner verifies source hashes, scene/state archives and exact identity replay
before starting. Each child executes the shared arm replay with finite timeout
and descendant cleanup. It preserves every candidate config, log, path rejection
and physical result; failures are not silently dropped. Worker implementation
snapshots must match the pinned replay/helper files. A changed implementation
cannot silently join a search. Inputs and outputs require no neighboring-project
runtime imports; model/data locations are explicit downstream artifacts.

`protocol.json` freezes limits derived from the original task/audit and the
development objective before evaluation. `progress.json` is a mutable progress
receipt; `search.json` is the completed immutable comparison. Full arrays, logs
and assets belong in the downstream private workspace.

## Ranking is separate from acceptance

Every eligible result must preserve the compiled scene, physics and controller
periods, reference/frame protocol, acquisition anchor, original phase cohorts,
full saved clock and all source controller settings except acquisition scheduling.
Shortened runs cannot obtain an objective, even if a malformed result claims
success. Original success/rejection lists are retained. A purported pass that
contradicts frozen task limits is rejected as inconsistent.

The fixed development objective is the sum of positive normalized task violations,
plus `0.25 * (1 - sustained_bilateral_hold_fraction)` and
`0.05 * normalized_hold_rotation`. It includes both wrist-frame drifts, suspended
clearance, raw-reference position/rotation errors and the original hold-rotation
limit. The sustained-contact term uses every saved reference-defined hold sample:
each thumb and at least two opposing pads per hand must reach half their requested
normal pressure. This uses privileged simulated forces, not independent contact
measurements or a force-closure certificate.

Actual task passes rank ahead of failures; physically unsafe results rank behind
safe failures. No low objective can promote a failed controller. Limits, force
targets and score windows are not tunable search inputs. New moving motion and
release remain unvalidated even after a stationary candidate is reproduced.

If a candidate passes, the runner freezes its selection and repeats its complete
stationary trial with unchanged settings. Promotion eligibility requires both the
original task pass and exact state reproducibility. A nominal repeat demonstrates
reproducibility, not generalization. If no bounded candidate passes, preserve the
baseline and investigate contact geometry/load paths rather than relaxing limits
or treating search score as control readiness.
