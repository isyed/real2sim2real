# Pad contact and closing-path planning

Optional `pad_planning` fits contact points and directions on compiled robot
collision pads. Legacy fingertip planning remains the default; physical limits
and grasp acceptance thresholds are unchanged.

`retargeting.pad_contacts.pad_frame` intersects a declared body-frame outward ray
with an owned collidable pad's convex hull. It records a body-local point, normal,
geometry name and assumed direction. Verify the direction against the robot.
This point is not the legacy tip site or a measured tactile contact.

`fit_pad_contacts` uses analytic point/normal Jacobians and collision residuals
in scratch MuJoCo data, preserving caller/object state. Fixture translation is
bounded to ±6 cm and rotation to ±0.35 rad, intersected with physical limits;
named bounds can only narrow them. Legacy fingertip IK uses ±0.45 rad rotation.
This reserve is part of the new planner, not a physical joint-limit change.

## Configuration

Add `pad_planning` with `pads` (one distinct `geom` and `outward_body` per target),
`opening_gap_m`, `preload_m`, `max_normal_error_deg` and `max_normal_force_n`.
Example pad: `{"geom": "thumb_pad_collision", "outward_body": [0, 1, 0]}`.
The existing anatomical force grouping still needs an appropriate robot adapter.
Opening gap is at most 20 mm, preload at most 2 mm, angle tolerance at most 45°
and the assumed normal-force ceiling at most 50 N. These ceilings are not
recommended or hardware-validated settings.

Set `rollout.close_offset_rad` to zero and `closure_search_rad` to `[0]`.
The stored `q_closed` is executed directly; extra scalar/finger closure feedback
is rejected because it would invalidate the checked path. Run the existing
bounded `object_grasp_pilot --phase plan` command.

Ray proposals are projected to object triangles with at most 0.5 mm adjustment,
recorded explicitly. Those triangles supply the target normals. Separate fits
produce contact, outward-open and inward-preload poses. Approach and closing
segments are sampled at at most 2 mm per slide and 0.02 rad per hinge (512 samples
per segment maximum), checking self/object/table collisions and joint limits.
This is sampled validation of a translated approach, not a general trajectory
optimizer or continuous collision certificate. Position/penetration criteria
remain active; normal, path and static-balance failures remain explicit.

## Static balance and finite contact area

`gravity_wrench_screen` solves gravity force/torque balance about the assumed COM
using an eight-direction inscribed friction pyramid. Initially each pad is one
point. The force ceiling is an assumption, not actuator feasibility. Feasibility
is not full force closure or dynamic success; failure of this conservative model
does not prove that no physical grasp exists.

`object_grasp_pilot --phase pad-patches` reassesses a pinned fitted plan without
changing joint poses, contact paths, mass, friction or force ceilings. Declare
`patch_reassessment.parent_plan`, `parent_sha256`, `radius_m` (≤5 mm),
`max_surface_gap_m` (≤1 mm), a fresh `planning_directory` and finite
`deadlines_s.pad-patches`.

Five rays per pad sample its actual compiled hull at the fitted pose. Retain a
point only when the object surface is nearby and normals oppose within 25°.
Every pad must retain evidence. Samples on one pad share one normal-force budget:
sampling more points cannot multiply available force. Other geometric failures
remain rejected. Sparse samples do not certify continuous contact or deformation.

## Explicit microhole repair

Optional `collision.microhole_repair` uses `retargeting.mesh_microholes` to cap
only isolated, consistently wound triangle/quad boundaries within a declared
metric diameter (ceiling 2 mm). Nonmanifold, larger or complex boundaries fail.
All source vertices/faces remain; the input asset is never overwritten. The audit
records source hash, boundary, diameter and added faces. Closedness and positive
volume are checked before inertia/decomposition. Repair is disabled by default
and cannot recover missing handles or certify proxy topology.

## Review and tests

`python -m real2sim2real.controller_actor_training.grasp_plan_review --config ...`
renders bounded static open/fit/closed hypotheses, including rejected ones.
Images state explicitly that they are not physics videos or successful grasps.

`tests/test_pad_planning.py` checks analytic Jacobians by finite differences,
known-geometry fitting, unchanged caller/object state, ownership, path collisions,
joint limits, gravity moments, shared pad budgets, nearby opposing patch evidence,
closure override rejection and bounded mesh repair. Synthetic kinematics do not
validate dynamic contact or hardware. Promotion still requires observed geometry
checks and free-object trials under unchanged physical gates.
