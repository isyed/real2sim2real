# Bilateral contact-force feedback

The bilateral fixture pipeline can regulate named pad forces through the existing
finger position actuators. This provides a bounded alternative to a fixed closing
command when a geometrically admitted grasp loses pressure under load.

Enable `contact_force_feedback` in a downstream `bilateral_evaluate` configuration
using `controller_profile: rigid_bilateral`. The adapter requires two admitted
five-pad hand plans, independent finger ancestor joints, unique unit-gear position
actuators, native joint limits and finite actuator force limits. The pressure-step
calculation is independent of the hand embodiment; the current pipeline binding
uses thumb/index/middle/ring/pinky collision pads on two wrist fixtures.

A representative configuration is:

```json
{
  "target_force_n": {"thumb": 3, "index": 1.5, "middle": 1.5, "ring": 1.5, "pinky": 1.5},
  "gain_m_per_n_s": 0.001,
  "max_normal_speed_m_s": 0.002,
  "max_joint_speed_rad_s": 0.2,
  "max_joint_offset_rad": 0.2,
  "damping_m2": 0.000001,
  "maximum_normal_gap_m": 0.008,
  "maximum_tangent_gap_m": 0.015,
  "close_activation_fraction": 0.75
}
```

These are controller hypotheses, not universal force recommendations or increased
actuator limits. Normal-force targets are capped at 10 N per pad by validation.
The adapter leaves physical joint limits, motor torque limits, the object and
contact geometry unchanged.

## Pressure control and observations

For each pad, measure solved contact normal force against the task object only.
Contacts against the table, another robot part or unrelated objects are excluded.
The signed force error specifies an outward-normal velocity: insufficient force
presses inward and excess force releases pressure. A damped least-norm normal
Jacobian maps this to finger-joint velocity. Joint stops remove blocked directions
before solving. Rate/travel limits and clipped offset feedback provide anti-windup.
Wrists and the free object are outside the controller's command ownership.
Controller telemetry records the solved force available when the command is
generated; post-step hold metrics are measured separately.

The default `normal_observation: planned` uses the admitted object's local normal
and contact label. Its geometry gates freeze corrections after excessive normal
or tangent separation. `normal_observation: solved_contact` additionally uses the
force-weighted measured contact position and object-outward normal whenever
force exceeds `minimum_observed_force_n` (default 0.1 N). Real named-pad contact
can then persist while the active patch moves. Loss of observed contact restores
the original planned-normal/gap-gated reacquisition behavior.

Both modes use **privileged simulator force and pose observations**. They are not
RGB-derived force measurements or hardware validation. Per-pad force goals need
not all be achieved: retain individual measurements, geometry gaps, offsets,
update/freeze counters and normal-source labels. An aggregate thumb/opposing-group
contact pass does not mean every finger maintains its goal.

An optional planned-surface tangent controller adds `tangent_gain_s_inv` and
`max_tangent_speed_m_s`. It combines normal pressing velocity and projected
surface-position error through a damped finger Jacobian, under the same joint
rate/travel/native limits. It is a hypothesis to test, not a default improvement:
the admitted geometric site can differ from the loaded contact patch, and trying
to recover that site can disrupt pressure or lose the grasp. Measured-normal mode
cannot be combined with this planned-site tangent objective.

## Stationary admission and moving-reference tests

Run the stationary condition first. A demonstrated-motion force-feedback run
requires `validated_stationary` with the successful result path and hash; optional
`protocol_sha256` pins its protocol. The stage verifies complete duration, states,
identical grasp/controller/scene settings and the original fidelity limits.
Only the input reference and output planning directory may differ. Failed holds,
changed gains, changed physical settings or relaxed reference criteria are rejected.
A stationary pass remains distinct from demonstrated-motion acceptance.

An explicit `physics_step_s` may refine the prepared scene timestep down to
0.25 ms for sensitivity checks. The original prepared physics receipt stays pinned
and each result records the actual `simulation_timestep_s`. Freeze that setting
as part of stationary-to-motion promotion.

## Lift-offset ownership

Optional `freeze_lift_at_hold: true` retains the achieved initial-lift offset once
hold starts. Demonstrated motion still uses the original object-pose increments
and native clock. This can prevent the clearance servo from adding an independent
vertical correction as the held object tilts: bottom clearance and object-center
reference position are different quantities. Apply the same ownership choice in
both stationary and moving tests, and retain the full physical/reference gates.

## Common object-pose feedback

Optional `object_pose_feedback` adds a bounded common rigid wrist correction after
the nominal hold/reference command. It uses the checked Cartesian fixture
contract in `wrist_coordination`; it is not arm IK. Specify
`translation_gain_s_inv`, `rotation_gain_s_inv`,
`max_translation_speed_m_s`, `max_rotation_speed_rad_s`, `max_translation_m`,
`max_rotation_rad`, and optionally `joint_limit_policy: project_common`.

The stationary goal is frozen at hold entry. In motion trials the desired pose
comes from the unmodified demonstrated object increments and native clock.
Corrections orbit both wrists about that desired object position, preserve their
commanded relative transform, and leave finger targets to the force controller.
All three translation axes can be corrected; this requires
`freeze_lift_at_hold: true` and excludes the older independent orientation and
fixed-world coordination loops. The latter retains its original XY-only behavior
when no explicit desired pose is supplied.

An accurate world-frame object pose does not establish grasp retention. Common
wrist rotation can reduce orientation error while increasing object-to-wrist
drift. Evaluate both wrists' relative drift, force/load support, clearance,
penetration and duration alongside position/rotation error. A failing stationary
candidate cannot advance to motion, and a passing reference metric cannot erase a
failed physical grasp. Simulator pose feedback remains privileged observation.

## Review and validation

Saved-state review assigns distinct media names when multiple studies use the
same candidate directory name. Baseline, stationary and moving videos must never
silently overwrite one another. Each section labels its motion condition.

Tests cover pressure direction, release, joint stops, anti-windup, force bounds,
object-specific force/normal measurements, command ownership, unchanged physical
state/limits, frozen promotion criteria and non-overwriting review names. Actual
free-object holds and motion trials remain separate required evidence.
