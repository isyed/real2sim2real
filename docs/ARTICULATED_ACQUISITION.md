# Mirrored contact layouts and articulated acquisition checks

A hand that fits at closure may have no valid path to that pose. Keep contact
layout, open/closed fitting, complete-hand insertion and restored-scene admission
as separate gates. These kinematic diagnostics do not establish retention.

## Conventions before optimization

Audit the actual compiled pad positions and normals at several native joint
postures. For mirrored hands, compare polar vectors with the declared reflection
matrix; do not assume a shared body-local outward ray needs its sign flipped.
Also do not negate joint angles indiscriminately: joint axes and local body
frames determine how each articulation mirrors.

An ordered thumb/index/middle target layout has chirality. Reusing its angular
ordering unchanged on the other hand can produce a poor fit even when the pad
frames and native flexion signs are correct. `articulated_acquisition.
opposed_taper_targets` takes explicit chirality and preserves finger identity
while mirroring the opposing fingers' angular order. Surface normals remain
outward; desired hand-pad normals are their negatives. All dimensions are metric.
Proposed contact locations remain hypotheses, not observations.

The optional `layout_roll_rad` trades the opposing fingers' angular separation
for vertical separation on the same tapered exterior. At zero it preserves the
original target layout; near ±90 degrees it stacks those contacts vertically.
This changes the wrist seed indirectly through actual pad-frame alignment; it
is not an instruction to rotate the wrist by that angle. Verify all generated
heights lie on the exterior. Moving a target does not validate the new contact
band's observed geometry.

Inspect discrepancies at zero and flexed postures. Small differences from
mesh-selected face normals are distinct from a large handedness/sign error.
A sampled convention audit does not validate every joint or hardware mounting.
Preserve older results when correcting a convention; label the corrected plan.

## Build and check articulated paths

Vary a bounded set of layouts and posture seeds with frozen acceptance limits.
Record open and closed target error, normal error, joint limits and collision
witnesses. A less-curled seed may avoid a local minimum without changing the
robot or increasing solver limits. Preserve all attempted fits, not only winners.

`insertion_waypoints` constructs explicit far-preshape, near-preshape (or a
partly shaped intermediate), opened and closed joint configurations. It accepts
caller-chosen postures and a bounded retreat in declared fixture-slide
coordinates. It does not infer a collision-free Cartesian approach.

Do not automatically straighten every finger. A fitted unused finger may need
to retain its posture: changing flexion while keeping abduction can cause it to
cross an adjacent finger. Preserve, neutralize or fold unused fingers as explicit
tested alternatives. Keep all original enabled collision geometry and native
joint limits; do not remove self-collisions to obtain a path pass.

`check_insertion_candidates` evaluates up to 64 complete paths using the existing
sampled full-hand checks. The entire path must satisfy the caller's penetration
limit, including palm, proximal and unused fingers. Endpoint-only checking is
insufficient. Selection by nominal penetration is a planning heuristic, not a
robustness or task-success claim.

## Restore the scene before a hold

Replay selected paths unchanged with every original obstacle and every required
uncertainty scenario restored. Verify source hashes, joint ordering, root poses
and replay arrays. When rotating a candidate around an object, rotate its root,
targets and slide coordinates consistently; recompute pad alignment after the
transformation rather than assuming equivalence for a multi-joint wrist fixture.

Report nominal isolated passes separately from uncertain-surface passes. A
nominal fit may penetrate an enlarged object; a smaller object may lose contact
even when the penetration check passes. Neither is verified reconstruction.

If the **closed** hand intersects the table, a different insertion path alone
cannot solve the problem. Replan wrist orientation and contact layout; change
contact height only where geometry evidence and protrusion exclusions permit it.
Do not shrink or move the table to admit a candidate. Only after complete
restoration and the remaining [control-readiness gates](CONTROL_READINESS.md)
should acquisition/lift/hold run. Completed-geometry mechanics remains explicitly
**conditional simulation**.

## Bounded agent proposals

Once a nominal complete-scene path exists, it can seed an agent-authored layout
proposal. Keep the simulator, geometry, collision links, uncertainty cases and
acceptance criteria fixed. Give the author structured open/closed alignment and
phase-specific collision witnesses, including body names and maximum penetration.
Permit only bounded layout, posture and insertion parameters; never execute
arbitrary proposed code or allow the author to redefine success.

A useful initial budget is one proposal followed by one feedback revision.
Compare both against the frozen baseline using the same fitter, solver budget
and full path checks. Treat already-inspected stress cases as development cases,
not held-out evaluation. If the budgets differ from the deterministic search,
report the difference and do not claim a general method winner.

The existing controller-policy and paired-grasp author contracts are distinct
interfaces. A tapered-container candidate needs a planning-only adapter before
an author call; do not reuse a graph's hold node to bypass the new geometry gate.
This is strategy proposal and evaluation, not residual RL or a trained control
policy. A nominal kinematic pass still does not admit hold tests when restoration,
surface uncertainty or contact-support prerequisites remain unresolved.
