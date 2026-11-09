# Tracking through pickup and placement

Inspect the visible object masks separately from its projected 3D pose. A mask
can keep following an object while a tracker leaves its 3D pose at the previous
height. A reported reliable-state count does not establish image agreement.
Use the reconstruction quality report and source-frame overlays before deciding
to regenerate segmentation or geometry.

Register clear frames on both sides of the failure and retain their provenance.
Tracker logs index the exported window locally. Translate those IDs through
`source_frame_mapping.json` before reporting original-video failures, especially
for reversed windows. A local failure at 14 in a window starting at source300
and running backward is source286, not source14. New trace records label this
scope explicitly; old traces remain interpretable using their retained mapping.
For nearly symmetric boxes, independent maximum-silhouette fits can select
orientations almost half a turn apart. Keep multiple admissible hypotheses.
`reconstruction_repair.anchor_path.select` selects among them with a declared
angular-rate cost. This is a selection prior, not an observation of the hidden
motion. It never admits a rejected hypothesis or certifies a whole reference.
Check intervening and held-out images, appearance, depth, support and joins.

An image bounding box is useful for initializing translation, but partial
occlusion or the wrong orientation can make a well-fitted box imply the wrong
depth. `silhouette_translation.propose` preserves the supplied rotation and
metric mesh; its optional `preserve_camera_depth=True` also freezes camera Z.
It returns only a bounded proposal, without image/depth/physical approval.
The unchanged native checks must still reject invalid candidates. An increase
in mean silhouette overlap can coexist with many depth failures, worse
previously good frames, floor penetration or large single-frame jumps.

For small local corrections with an already qualified orientation,
`LocalRegistrationConfig(preserve_rotation=True)` fits translation without
rotational drift. Its default remains false. All previous native acceptance
limits remain active; a fresh initial fit still needs independent renderer
validation. Do not replace a trusted anchor with a later bounding-box fit merely
because the latter has a larger image score. Keep locked anchors exact and
validate every complete-clip seam after temporal smoothing.

Inspect the depth measurement itself too. A sparse set of convex-hull vertices
can include rear surfaces and have fewer visible samples than the original
visual mesh. Front-facing samples can change the measured error but may have
insufficient support. Keep both diagnostics, enforce the sample-count limit and
use the intended mesh/renderer for qualification. Do not hide contradictory
visible depth or lower the acceptance thresholds to approve an attractive video.

Use `comparison_video.render` candidate `uncertain_frame_ids` to label existing
but unqualified poses in the video and bind that ledger into its cache receipt.
The separate partial-prefix mode still shows unavailable geometry as unavailable.
Showing a complete video does not imply that every pose was observed or accepted.

After a track qualifies, rebuild paired retargeting/contact phases and evaluate
the frozen controller. If the robot falls before its scheduled hand phases,
finger-closing changes alone cannot establish a grasp. Separate reaching,
balance, contact timing and contact retention. Do not treat a better image fit
as actor-training readiness or a successful physical grasp.

A continuously validated prefix can support a separate short first-grasp
diagnostic while full-clip recovery remains blocked. Slice the original and
measured tracks to the identical native time interval, reuse the identical
human conversion and rebuild each paired canonical mesh/pose bundle. Recompute
contact phases from those references and run initial-placement, isolated
settling and unassisted physics checks. Report physical contact separately from
geometric contact labels. Prefix scores against different object targets are
not competition-score improvements or evidence of complete task execution.

`controller_actor_training.collision_proxy.package` creates an explicit
collision-only convex approximation of a portable robot reference. Visual
geometry, recorded motion, support and inertia remain unchanged, and its
receipt declares that concavities are filled. This opt-in hypothesis still
requires intended-solver settling and robot physics; fewer collision triangles
alone do not certify stability. Render before/after canonical bundles separately
before composing their physical panels so one policy's mesh frame is never
silently used for the other policy's trajectory.
