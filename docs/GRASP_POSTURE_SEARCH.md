# Contact placement and wrist travel search

`controller_actor_training.grasp_posture_search` proposes small changes to an
existing bimanual grasp, then compares actual force-limited physics on matched
material cases. It preserves the original baseline and rejects an apparent
improvement if any required development case fails. It builds on
[common wrist coordination](WRIST_COORDINATION.md).

## Inputs and contract

The configuration retains the prepared scene's `output` and `planning_directory`.
`posture_search.output` names a fresh separate experiment directory.
`parent_trial` and `parent_trial_sha256` pin a successful retained trial and its
state archive. Robot definitions, physics, tabletop/object assets, approach/lift
timing, closure and existing feedback must match the parent. Hold duration may
increase. The support contact plan and hand base frames remain fixed.

Declare at most six `variants`, each with a distinct local `name`, anatomical
`finger_offsets_m`, and `joint_bounds`. Offsets select real local opposing surfaces
through the existing surface-ray routine. Bounds may only narrow the robot's
physical and existing IK ranges; they cannot widen limits or move the fixture's
zero frame. For example, a positive initial wrist angle can reserve negative
travel needed during lifting, at the cost of positive travel. A bound is an IK
constraint, not a promise about the subsequent dynamic state.

`max_nfev` bounds contact fitting (at most 400). The primary hand is refit while
the second hand remains at its existing parked pose. The original approach
direction is checked with samples at no more than 2 mm spacing. Contact-error and
penetration thresholds are inherited without relaxation. The support insertion
was planned for the old primary lift; its collision safety after a changed grasp
must be tested dynamically. Geometric admission alone never validates it.

`retargeting.grasp_kinematics.fit_contacts(..., joint_bounds={...})` exposes the
same restricted-range fit independently of this campaign runner. Bounds must
name robot-owned joints and be finite, ordered, nonempty subranges. The default
fit is unchanged for existing users.

## Bounded stages

```bash
python -m real2sim2real.controller_actor_training.grasp_posture_search \
  --config experiment.json --phase plan
python -m real2sim2real.controller_actor_training.grasp_posture_search \
  --config experiment.json --phase compare
```

`deadlines_s.plan` and `deadlines_s.compare`, under `posture_search`, supply finite
CPU budgets. The supervisor hides GPUs and limits math threads. The comparison
uses the generated plan and checked protocol, and runs at most six named
`cases`, with `mass_scale` and `friction`, for every admitted proposal.

Two controls use the unchanged parent grasp: `retained-baseline` with its original
feedback and `shared-baseline` with the specified common wrist controller.
Proposed grasps use that same common controller. These matched controls separate
contact/posture changes from the earlier change in feedback. Standard task,
contact, penetration and both-wrist drift gates remain active; the common
controller retains its independent world-drift and command-consistency gates.

Reports include geometric rejections, complete trial failures, material/timing
settings, actual loaded finger groups, independent world drift, and minimum
distance to each wrist's actual angular limits over saved position samples.
Negative margins remain negative; sampled margins do not certify unsampled
intervals. State archives and result/scene hashes preserve replay provenance.
The original scene, plans and outcomes are not overwritten.

Only a new candidate passing every declared case can be selected. That selection
is a development result requiring frozen longer and perturbation validation.
It does not establish arm reachability, deployable sensing, policy training,
human-reference fidelity or hardware success. A failed candidate can remain useful
as a diagnostic without replacing its baseline.

Tests in `test_grasp_posture_search.py` cover restricted IK bounds without changes
to physical limits, ownership/invalid-bound rejection, negative margin reporting
and rejection of partially successful candidates. Actual contact improvement
requires the matched physics trials.
