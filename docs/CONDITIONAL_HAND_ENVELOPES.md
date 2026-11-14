# Conditional surface envelopes and full-hand fitting

Use this stage when an observed surface is incomplete and a completed shape is
needed to investigate a grasp. Label the result **conditional simulation**, not
verified reconstruction. Kinematic feasibility, observed-contact admission and
loaded retention are separate decisions.

## Evidence and hypotheses

1. Inspect native stereo masks before changing the shape. Save reviewed
   replacements separately, with frame IDs, pixel coordinates, erosion and
   source hashes. Report removed pixels and local depth residuals. An
   assistant-reviewed mask is not independent human ground truth.
2. Localize visible protrusion mounting patches in both native views. Record
   vertical correspondence disagreement and rejected pairs. A mounting center
   does not measure the protrusion's tip, full extent or hidden attachments.
3. Declare surface and obstacle uncertainty explicitly. Separate measured
   disagreement, annotation sensitivity and assumed engineering margins.
   Perturbing annotation-box corners is a sensitivity study, not a guaranteed
   bound or calibrated confidence interval. Keep unobserved sectors unknown.
4. Preserve the frozen comparison. Build separately named nominal, expanded and
   contracted hypotheses. Convex obstacle envelopes can fill real openings and
   reject feasible access; they do not certify unseen space.

## Reusable kinematic interface

`retargeting.conditional_hand_scene.prepare` retains a fixed-root robot's
original articulated collision links and adds bounded Cartesian wrist joints.
It accepts explicit mesh paths, spherical exclusions and solid cylindrical
exclusions, and records geometry hashes. Existing outputs cannot be overwritten.
Objects remain fixed: this is an IK/collision scene, **not a loaded physics
scene**. Heavy MuJoCo imports are confined to the collision stage.

`align_pad_seed` initializes a proper rigid wrist transform from actual compiled
pad positions and normals. It does not solve finger joints or establish contact.
Use it to check whether a poor wrist seed is dominating a local fit; retain
results from both initializations rather than silently replacing a failure.

`sample_paths` interpolates complete-hand joint waypoints and reports collision
witnesses against the cup, table, rack, protrusions and self. Include every
enabled owned collision geometry: palm, proximal links, distal links, pads and
unused fingers. Preserve model collision exclusions and disclose their scope.
The routine uses scratch state and performs no dynamics steps.

Default sample spacing is at most 2 mm for slide joints and 0.02 rad for hinge
joints, with a finite per-segment budget. This is sampled joint-space collision
checking, **not continuous swept-volume certification or a positive-clearance
measurement**. Joint interpolation is not a Cartesian straight-line guarantee.
Review palm/proximal witnesses explicitly; checking pad alignment alone is
insufficient.

Fit nominal geometry, then test the **same path without refitting** against the
uncertainty scenarios. Retain pad-position error, normal error, joint limits,
penetration, contact gaps and the offending body/geometry/path fraction. Report
worst-case and per-scenario results, not an averaged pass rate that hides a
failure. Caller-owned acceptance limits must remain frozen.

## Decision boundary

Failed local fits reject those candidates, not the entire manipulation task.
If large target errors coexist with obstacle collisions, first separate
hand/cup fitting from scene access using a clearly labeled isolated-cup
diagnostic. Restore all obstacles and uncertainty envelopes before admitting
an approach. Do not remove an exclusion merely to obtain a pass.

For a matched isolation experiment, retain the cup, all hand collision links,
self-collision exclusions, pad frames, seeds, target layout, solver budget and
acceptance limits. Remove only named environment obstacles in a separate scene.
Record open and closed alignment separately from maximum path penetration: a
nearly acceptable closed grasp can still have an invalid pregrasp or insertion.
Replay the prior path in the isolated scene as well as fitting a new isolated
candidate; this distinguishes changed collision cost from changed initialization.

Restore the exact source obstacle scenes and replay each new path without
refitting. Compare source hashes, enabled geometry and joint ordering. A failed
isolated candidate may be replayed for diagnosis, but cannot gain admission from
that replay. `assess_obstacle_restoration` requires alignment, isolated path
acceptance and every explicitly required restored scenario; missing scenarios,
nonfinite errors or any failed limit reject the approach. The helper aggregates
caller-supplied audits, so callers must verify scene completeness and unchanged
paths. It never promotes observed geometry or admits physics.

A radial translation of an already opposed grasp can sweep the far-side
fingers through the object even when the final pose has little penetration.
Plan opening and insertion as articulated motion, with intermediate full-hand
collision checks. Left/right discrepancies also warrant checking mirrored pad
frames, joint conventions and seed posture before concluding physical infeasibility.

Use [articulated acquisition](ARTICULATED_ACQUISITION.md) to audit target
chirality and construct bounded multi-stage finger/wrist candidates without
silently straightening unused fingers or admitting endpoint-only fits.

Only after full-hand fitting, insertion checks and basic arm reachability should
the appropriate acquisition/lift/hold gates run. A fixture hold still requires a
later arm-mounted hold. Completed-geometry mechanics remains conditional even
if successful. See [control readiness](CONTROL_READINESS.md) and
[native contact evidence](NATIVE_STEREO_CONTACT_EVIDENCE.md).
