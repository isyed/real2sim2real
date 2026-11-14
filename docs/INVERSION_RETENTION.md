# Retiming inversion and coordinating pad forces

Preserve the scene, actuator limits, free-object dynamics, acceptance thresholds
and original controller before changing motion or contact control. Keep a complete
nominal task attempt distinct from a successful task or acceptance for control.

## Changes that can be isolated

`controller_actor_training.phase_timing.PhaseTiming` maps strictly increasing
reference timestamps to explicit execution timestamps. It can lengthen one phase
while preserving earlier timing and shifting all later events coherently. There
are no repeated reference knots that insert an implicit pause. Time queries clamp
to the endpoints. Callers must audit velocity at phase boundaries: a piecewise
linear clock does not by itself make an arbitrary trajectory differentiable.

Retiming preserves the geometric path, but changes acceleration, controller
integration time and contact response. It therefore still needs a full physics
replay. A different orientation interpolation between the same endpoints changes
the path itself and needs new arm IK and dense interpolation/collision checks.
For a half turn, the two rotation directions can have very different loaded
contact behavior even with identical final poses.

When testing a later segment, copy the accepted prefix timestamps and joint
commands exactly. Re-solving IK or changing interpolation density before the
intended change can alter contact dynamics and confound the comparison. Check
the replayed state/command prefix as well as the geometric endpoint agreement.

`PadPressureCorrection.update` accepts optional per-pad `targets_n`. Equal force
per finger is not necessarily balanced across sides of a grasp: one thumb
opposed by two fingers may need its target compared with their combined force.
Target balancing is a heuristic, not a full wrench solution. Gravity, differing
normals, friction, moment arms and joint authority determine actual force sharing.
Targets remain bounded to 5 N per pad; original native torque limits are unchanged.
The quantity is force in newtons, not pressure in pascals.

The normal is perpendicular to the contacted surface. The existing
`contact_force_control.object_pad_observations` provides force-weighted current
contact points and object-outward normals. These can replace a planned normal
when sufficient object contact is observed, with a declared fallback on contact
loss. A moved contact should not be treated as if it still used the original
surface normal. These observations are privileged simulator data, not a real
robot tactile estimator.

Optional `tangent_tasks` compose the existing normal force step with bounded
material-point retention, using `ContactAnchors` and contact-point Jacobians.
Each pad gets either no tangent task or a projected 3-by-N Jacobian and a world
velocity capped at 5 mm/s. Only the pad's declared joints are used. Normal and
tangent tasks share the same joint-rate and offset limits; no extra actuator
authority is created. Omitted tasks preserve the archived scalar-normal behavior
exactly. A contact-anchor variant can perform worse when it consumes correction
capacity needed for pressing; evaluate rather than assume improvement.

## What to measure

Record reference and execution times, exact scene/path identities, per-pad force
targets and actual forces, correction offsets, object position/orientation,
release state and collision witnesses. Retain qpos, qvel and commands for replay
and contact inspection. A zero-change replay should reproduce the frozen states
and commands before comparing altered timing or feedback.

Check task-specific orientation independently of full rotation error. For an
inverted container, axis tilt measures whether the opening points down; axial
spin may have a different task relevance. Keep both if comparing reconstruction
or reference tracking.

Do not stop interpretation at successful inversion. Supported opening can leave
residual proximal-link contact; removing that contact may expose an unstable
rack bearing configuration. If the object tips after the hands exert no force,
increasing grip pressure is no longer a direct solution to the final placement.
Reassess the local bearing pose and release/withdrawal path in the retained scene.
A full-duration rollout with the wrong final orientation remains a failed task.

A small placement-pose adjustment can change which rack surfaces bear the load
without changing reconstruction or acceptance limits. Check it with the full
hand and arm, then replay acquisition through release from the original initial
state. Preserve exact repeat evidence and add settling time when earlier trials
showed delayed tipping. A nominal task pass with unresolved quality warnings is
still distinct from robustness or control acceptance.
