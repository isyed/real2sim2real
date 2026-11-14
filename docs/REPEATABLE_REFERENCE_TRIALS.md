# Repeatable boundaries, scene evidence and reference evaluation

A repeatable trial has a pinned initial simulator state and controller state,
fixed start/end joint targets, explicit capture and execution clocks, and a
declared completion gate. Returning the robot home does not reset the object;
the next independent simulation trial must restore the entire initial scene.
Hardware resetting is a separate operation.

`controller_actor_training.joint_boundary_path.JointBoundaryPath` supplies
unwrapped joint interpolation with zero velocity at its waypoints and an analytic
maximum joint speed. Screen the interpolated full arm and hand against all scene
obstacles before use. `return_admitted` checks supported placement, complete
opening, low hand contact and low object speed. It does not certify clearance.
Use robot reachability to select a home region resembling the visible human
wrist; human arm joint angles need not be copied across embodiments.

Preserve an admitted core controller while testing approach/return boundaries.
New initialization changes dynamics, even if the core commands are unchanged:
run the complete free-object sequence and repeat it. Report home position,
orientation and finger errors separately from task and control acceptance.
Exact deterministic repetitions establish repeatability, not robustness rates.

## Local geometry and display

`egocentric.local_plane` fits a bounded support-plane hypothesis and assesses all
reserved observations without trimming validation residuals. Define the visible
surface mask before fitting; floor, ribs and rim are distinct surfaces. A plane
fit on a small textured patch does not establish the entire placement region.
Rejected fits never replace their source automatically.

Trace actual object-support forces to named collision surfaces. A plausible
floor may carry no load while a few inflated observation cells carry the object.
Changing an unloaded surface and reproducing the same result is a negative
sensitivity check, not evidence of improved physical geometry. Inspect the
loaded cells against images and reconstruct connected bearing surfaces; unknown
space can remain a planning exclusion without becoming a force-producing solid.

`retargeting.collision_visualization.CollisionRenderer` optionally accepts
`static_surfaces=[(world_triangles, rgb)]` and `hidden_geoms`. This supports a
separate observed-surface view, retaining holes and concavities instead of drawing
convex collision hulls or inflated cells. These options change display only.
Keep an explicitly labelled collider view alongside it. Neither a smoother
picture nor a successful rollout validates the reconstruction.

## Evaluation and subsequent proposals

Freeze controller outputs, object meshes in their pose body frames, units,
timestamps and source hashes before evaluation-only reference data is read.
Reference scans/poses must not feed reconstruction or controller inference.
Preserve invisible/missing-frame coverage and declare object symmetry and mesh
registration conventions.

Distinguish capture-clock tracking from duration-normalized comparison. Retiming
the whole simulated sequence to fit a video can diagnose spatial behavior, but
is not evidence of matching the demonstrated timing. Include the static object's
contribution when explaining aggregate object metrics. Task completion, reference
tracking success and robustness success rate are separate quantities.

For ASL or other proposal generators, first freeze the scene, reference clock,
home contract, controller bounds and evaluator. Verify a pass-through proposal.
Use a bounded proposal/revision budget; forbid geometry, collision and acceptance
changes inside controller proposals. Do not supply evaluation-only reference
geometry/poses to the author. Residual learning comes after the same interface
reproduces the chosen frozen controller with zero corrections.

Tests: `test_joint_boundary_path.py`, `test_local_plane.py`, and
`test_observed_surface_render.py`.
