# Contact support, clearance and surface uncertainty

Do not rank grasp candidates by penetration alone. Moving pads away from a
surface can improve clearance while removing the grasp. Keep the previous
geometric verdict and add an explicit contact-support diagnostic before
interpreting a candidate as a better acquisition or hold seed.

## Reusable finite-sample evidence

`retargeting.contact_uncertainty.convex_pad_samples` ray-projects a central
cross onto the actual compiled convex pad hull. It uses hull planes and needs
no optional spatial index. It preserves missing intersections instead of
inventing contact area. Pass explicit body-frame point and normal conventions.

Transform those samples and their face normals to the shared world frame.
`tapered_sample_evidence` measures signed side-wall distance and opposing-normal
agreement against an explicitly declared tapered-container hypothesis. It checks
wall height; end caps and hollow interiors are excluded. The lateral distance
uses the ideal taper, rather than claiming exact agreement with every facet of
a tessellated collision mesh. Record this representation difference.

The diagnostic accepts explicit proximity, penetration and angle limits. A
sample counts only if it is on the wall band, sufficiently near the surface,
not too deeply penetrating and facing the surface. Require evidence for every
intended pad. Five samples are not a finite contact-area certificate, independent
image evidence, friction/torque feasibility or physical retention.

If using a separate collision-distance API, validate its witness segment and
result consistency. Do not interpret a zero return as contact when its returned
nearest-point segment is inconsistent. Keep unavailable distances unavailable;
an analytic side-wall diagnostic must be labeled as a different measurement.

## Separate uncertainty factors

Replay the same frozen path in nominal, object-size-only, table-height-only,
obstacle-only and combined stress scenes. Keep every obstacle present. Record
both full-path maxima and endpoint evidence; do not compare an endpoint result
with a historical full-path peak without labeling the difference.

Use the same acceptance limits. Fit-weight or planning-scene changes must be
declared new conditions and all candidates retained. Outward target offsets
must still be evaluated against the actual surface, not only the offset target.
A smaller-object penetration pass can be contact failure.

## Bounded geometric support-stop diagnostic

`closure_stop_decision` rejects invalid observations, penetration violations and
joint-limit failures before accepting support on every required pad. This small
decision function can audit a fixed precomputed closure sequence across surface
hypotheses. It does not command the robot or simulate forces.

Freeze the sequence and approach before testing conditions. Do not refit the
closure per condition and call it one controller. Distinguish approach acceptance
from closure acceptance, and record the first rejected sample and offending link.
Exact simulated surface geometry is a privileged signal; a positive result is
only an upper-bound kinematic diagnostic until sensing, dynamics and contact
forces have been validated. Never claim a sensor-based policy or robust hold
from this test alone.

If a proximal link hits the object before pad support appears, a stop rule or
more pressure cannot repair that path. Replan proximal posture and approach
clearance while retaining the improved nominal pad seating. Apply the full
[control-readiness gates](CONTROL_READINESS.md) before holds and training.

## Protect proximal links without losing pad seating

`retargeting.collision_reserve.CollisionReserve` adds a selected-link collision
cost from an explicit uncertainty scene to `fit_pad_contacts`. For example,
reserve a proximal phalanx against an enlarged object while fitting the pad
frames in the nominal scene. Pass the reserve through `collision_reserves`.
Every primary collision cost remains active; this is an additional objective,
not an exclusion or a safety certificate.

The context uses scratch state and must have the same owned joint ordering,
embodiment and root/world transform as the primary scene. Ordering is checked;
physical equivalence and frame alignment remain the caller's responsibility.
Use complete-hand collision checks in every required scene after fitting. A
cost on one link can move the failure to a palm or thumb link.

`normal_scale` optionally strengthens the unit-normal residual. The default
0.35 and an empty reserve sequence retain the historical fitting objective.
Its bounded range is 0.1–0.35; it weights a unit-vector difference and is not an
angle limit. Position weighting, normal weighting and selected-link collision
weighting must be recorded independently from acceptance thresholds.

Audit opening, insertion and every executed closure sample. A stop with nearby
opposed pad samples can still violate the separately declared pad-frame normal
limit. Check that limit along the closure prefix, not just at the stop. Keep
all required gates conjunctive and preserve slight threshold failures without
rounding them into passes. Do not treat a supported endpoint after an unsafe
approach as an admitted grasp.

Preserve fixed-pose and adaptive-closure comparisons separately: their stopping
states, paths and maxima differ. Report any loss of the previous table margin,
even when both candidates remain within the penetration limit. A maximum
penetration check does not establish positive minimum clearance. Geometric
support-stop feedback using exact hypothetical surfaces remains privileged
simulation evidence, not tactile sensing or force/retention validation.

## Joint reserves and path-wide orientation rejection

Reserve palm/thumb links together with a troublesome proximal finger: a cost on
one link can otherwise transfer the collision to another link. The optional
`normal_guard_deg` in `fit_pad_contacts` adds a strong penalty outside a declared
normal cone. It does not guarantee feasibility. Keep this planning margin below
the independently declared acceptance angle and audit the resulting path.

`closure_stop_decision(..., frame_normal_errors_deg=...)` can reject a sampled
pad-frame orientation violation before returning `stop_on_support`. The vector
must contain one finite angle per intended pad. Call this rule on every executed
closure sample, not only at the endpoint. Existing callers that omit the vector
retain their earlier rule and should not be described as enforcing frame angles.

A pass across geometry hypotheses is a sampled hand-path result. It does not
validate physical object/environment initialization or turn conservative keep-out
volumes into measured contact material. The free-object preflight described in
[control readiness](CONTROL_READINESS.md) is a separate gate.

Terminology: `stop_on_support` is a historical API/result label for **geometric
pad proximity with opposing normals**. It does not mean measured force support,
weight bearing or retention. User-facing reports should explain this label or
say “stop on geometric pad seating.” Likewise, joint palm/thumb/index planning
means coordinating wrist and finger posture to seat pads while clearing the palm
and proximal links; it does not prescribe palm contact.

## Physical contact material from local observations

`observed_colliders.occupied_boxes` creates bounded occupied cells covering every
supplied metric point. It preserves gaps between disconnected cells instead of
bridging them with one global convex hull. Thickness is an explicit assumption;
empty cells are unobserved, not certified free. Do not filter observations using
the desired grasp success or object-intersection outcome.

`coverage_upper_bounds` measures distance to nearby occupied cells for separate
validation views. Report coverage and its spatial limitations. Reused stereo
views are not an independent measurement of hidden surfaces.

`apply_contact_roles` modifies a separate XML tree: original conservative volumes
remain named planning-only geometry, while occupied boxes supply physical rack
contacts. This function does not itself enforce planning exclusions. The caller
must continue to check the robot against them, including the actual moving object
transform during dynamics, and must audit free-object initialization again.

A force-feedback controller can respond to contact variation without establishing
robustness by itself. Distinguish geometric size scenarios, privileged simulator
force feedback and validated hardware sensing. Freeze a controller before testing
size/material/pose perturbations; record failures and avoid refitting each case.
Sustained contact at seating only admits a lift attempt. It is not proof of enough
vertical support or a complete hold.
