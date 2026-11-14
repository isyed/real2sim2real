# Nominal arm and scene diagnostics

An exploratory full task can precede broad robustness testing or a strict hold
pass. Preserve the old strict comparison, declare the changed geometry and path,
and distinguish task completion from acceptance for robot control. The experiment
must still simulate a free object, physical collisions and limited actuators.

## Adequate local geometry

Use measured exterior dimensions where available. Model openings explicitly:
one convex collision mesh for an entire hollow container fills its opening and
can create fictitious support during inversion or placement.
`retargeting.container_colliders.tapered_shell` returns separate convex wall
sectors and a bottom. It uses metres, a bottom-centred origin and a +Z axis.
Wall thickness is radial; wall/base thicknesses are supplied assumptions unless
independently observed. The pieces have outward faces. Base/wall overlap means
their volumes must not simply be summed to infer mass or inertia.

Keep three scene roles explicit:

- Observed physical surfaces, with observation provenance and uncertainty.
- Completed physical surfaces, declared as hypotheses rather than observations.
- Planning exclusions, which report clearance uncertainty but exert no force.

A small support completion can make an end-to-end experiment useful without
reconstructing every distant surface. It must not erase inconvenient observed
obstacles. Check local support height and footprint, hollow entry clearance and
the full hand. A successful completed-scene simulation remains conditional on
those assumptions; it does not validate hidden rack prongs or wall thickness.

## Connected arm planning

Endpoint IK passes do not establish a usable path. Continue a consistent IK
branch through acquisition, lift, inversion, transfer, supported placement and
withdrawal. Check the actual interpolation used by the controller, including
full fingers, palm, adapter, forearm, arm self-collision and object/environment
contacts. Coarse samples can miss narrow obstacles or create interpolation
errors even when each IK endpoint is accurate.

Pin base pose, mount assumptions, native joint limits, velocity limits, torque
limits, path timestamps and source hashes. A relocated simulated base also needs
a physically suitable support; it is not evidence of a verified hardware mount.
Collision checks are finite sampled checks, not a continuous collision proof.

Inspect loaded tracking independently of the grasp. A mechanically intended
base/shoulder interface may have overlapping collision meshes that accidentally
constrain joint motion in an imported model. Do not respond by increasing torque
or disabling arm collisions wholesale. `joint_interface_collision.exclude_joint_interface`
adds one explicit, named exclusion after a topology check. The caller must
justify the interface; the helper rejects world and multi-joint pairs, and other
contacts remain enabled. Recompile and repeat the dynamics, retaining the failed
import as a model-integration diagnostic rather than a grasp-performance result.

## Diagnostic continuation

`controller_actor_training.diagnostic_verdict.DiagnosticVerdict` defaults to
stopping on warnings. An explicit `continue_quality_warnings=True` records their
first times while permitting continuation. Physical and numerical failures
remain latched hard stops. The caller owns the classification and must pin the
warning and hard-stop thresholds in its experiment protocol.

The verdict separately records:

- `complete_horizon`: the requested time interval ran.
- `task_completed`: externally evaluated task conditions held, with no hard stop.
- `strict_quality_pass`: task completed without recorded quality warnings.
- `control_accepted`: always false; this diagnostic cannot certify control.

For a placement task, task completion should require persistent object support,
appropriate orientation/location, low residual motion and fingers actually
released. Reaching a wrist waypoint or playing the complete animation is
insufficient. Keep release dependent on measured support and prevent withdrawal
while the fingers remain closed. Existing `finger_release.FingerRelease` provides
support dwell and a one-way opening latch; its `table_*` observation fields can
refer to a declared rack/support surface selected by the downstream experiment.

After reasonable nominal completion, test bounded correction authority and, if
useful, residual learning. Later robustness and control admission still require
payload/friction/geometry variations, sensing and timestep checks, appropriate
physical limits, and arm-mounted retention. See [control readiness](CONTROL_READINESS.md).
