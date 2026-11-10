# Closing commands, surface feedback and contact evidence

`controller_actor_training.finger_closure` separates geometric fingertip fitting
from the command used to close a physical hand. It extends the existing frozen
grasp comparison; it does not change object geometry or redefine success.

## Named closing offsets

The legacy command adds `rollout.close_offset_rad` to each supported finger
flexion joint, then clips to physical joint limits. Optional `finger_closure`
overrides individual offsets:

```json
{
  "offsets_rad": {
    "right_index_MCP_FE": 0.3,
    "right_index_PIP": 0.1,
    "right_index_DIP": 0.05
  }
}
```

Keys must name flexion joints owned by the candidate hand; unknown joints,
wrist coordinates, nonfinite values and magnitudes above 0.6 rad are rejected.
Unlisted flexion joints retain the default offset. Negative offsets can explicitly
unroll a distal joint; they do not change its physical range. Approach, timing,
primary/support grasp plans and all actuator limits remain separate inputs.

Every new trial records `closure_target_geometry`: scratch forward kinematics of
the fitted pose and the closed command, including fingertip displacement and
distance from the planned targets. This never assigns physical coordinates.
Large command displacement is evidence that the closing command differs from the
geometric fit, not evidence that real fingers move through an object or that a
particular contact force will result.

## Optional contact-aware finger feedback

`finger_surface_feedback` declares explicit groups, each with a force-group
`name`, a declared fingertip `site`, and its independently actuated ancestor
`joints`. Supported force groups are thumb, index, middle, ring and pinky. The
first declared robot contact site/target belongs to the thumb. Root/wrist joints,
shared joints, non-ancestor joints and duplicate groups/sites are rejected.
Different robot anatomies require an explicit site/joint/contact-group adapter.

Required bounded settings are `gain_s_inv`, `max_speed_rad_s`, `max_offset_rad`,
`target_force_n`, `force_gain_m_per_n_s`, `max_inset_m`, and `max_outset_m`.
Limits are respectively 10/s, 2 rad/s, 0.8 rad, 10 N, 0.01 m/(N s), and 5 mm
in either normal direction. These ceilings are implementation bounds, not
recommended gains or validated hardware settings.

The controller converts the original contact points into the object's local
frame. During closing, lifting and holding it transforms them using the observed
object pose. Signed normal offsets integrate the difference between target and
measured per-finger normal force. Damped site-Jacobian corrections move only the
declared finger joints. Each group's joint-speed norm and each joint's integrated
offset are bounded; physical limits clamp the resulting commands with anti-windup.
The thumb uses the opposite surface normal from its opposing fingers.

The current normal approximation comes from the selected surface pair's closing
axis. This is a contact-target controller, not optimization over actual pad
surface orientation or frictional wrench feasibility. It may saturate, fail to
reach the target, disturb the grasp, or retain contact only intermittently.

Object pose and the previous integration step's contact forces are privileged
simulator observations. No object coordinates, applied forces, welds, collision
meshes, physical limits or acceptance thresholds are changed. Scalar
`contact_feedback` and per-finger feedback cannot run together. Common wrist
coordination and the support load loop remain distinct controllers. Deployment
requires real state/contact estimation and independent validation.

## Evidence and comparison

The existing bounded `grasp_robustness` CLI accepts both new controller settings.
All declared controllers use the same pinned grasp, geometry and material cases;
only an all-case development pass can enter its predeclared longer validation.
Frozen validation and the posture-search source contract reject changed closing
or finger-feedback settings. An unsuccessful controller remains a diagnostic.

Optional `contact_patch_logging` records actual normal force by primary robot
geometry at each saved sample. Unnamed geometries retain their ID and owning body
in the label. Geometry-level logging does not silently change the established
anatomical contact-group scorer. Existing hold records report each group's mean
normal force and active sample fraction. Reviews display these observations
alongside the unconstrained command audit. A large palm/thumb normal force does
not imply opposing-finger support; normal force is not vertical load share.

The common physical gates remain active: lift clearance, sustained opposing
contact, both-wrist drift, support load, orientation and penetration. Independent
world drift and common-command consistency remain additional gates when shared
wrist control is enabled. Grasp points being close in space is not a pass.

## Numerical integrity

`simulation_health.step_rejection` rejects solver warnings and nonstandard or
reset simulation clock increments. MuJoCo may reset after invalid accelerations
and leave finite state values; checking finiteness alone is insufficient. Trial
reports record warning counters, requested duration and the actual final clock.
A stopped trial cannot be promoted through otherwise small partial-hold errors.
Old results are preserved, with any retrospective clock audit stored separately.

`test_finger_closure.py` covers named offsets/ownership, frozen controller fields,
scratch command audits, finger-only bounded tracking with fixed object/wrist
coordinates, inward/outward force feedback, and numerical clock/warning rejection.
The optional MuJoCo test uses ideal joint tracking solely to test kinematics;
free-object physical success requires its own bounded experiments.
