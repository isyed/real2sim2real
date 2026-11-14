# Frozen geometry and native-stereo contact evidence

Keep shape plausibility, finite contact support and robot access separate. A thin
strip of consistent stereo depth can constrain radius without observing enough
area for a finger pad. Low residuals on that strip are not evidence of an observed
opposing grasp or a clear insertion corridor.

Use native rectified stereo points and native image cues to remove third-camera
registration from this check. Freeze dimensions and calibration. If rigid pose
must be fitted, use only declared fit pixels and cue uncertainties, and retain
other pixels for depth checks. Held pixels remain correlated stereo observations,
not independent ground truth. Distinguish original shape-fit views from later
checks and explicitly unavailable views.

`egocentric.native_contact` provides:

- `side_coordinates`: normalized axial height and signed tapered-wall residual
  for fixed dimensions.
- `tangent_evidence`: local footprint counts, occupied grid cells, normal offset,
  fitted-plane residual and normal agreement. Missing evidence stays unavailable.
- `place_distal_sweep`: place an actual compiled pad/distal-link point set at a
  proposed contact and translate it through explicit withdrawal distances.

The latter is a necessary geometry screen. It does not solve finger joints,
include proximal links/palm/arms, or certify continuous swept-volume collision.
Vertex sampling can miss face/edge collisions. Unknown stereo visibility is not
free space, and a point behind a measured surface is not necessarily occupied.

`retargeting.pad_contacts.convex_outward_point` exports a ray-selected point and
normal from compiled convex geometry using Qhull planes. It uses the convex
vertex centroid and needs no optional ray-tree index. It does not alter the
existing `pad_frame` fitter or its volume-centroid ray convention. Export the
pad point/basis explicitly; never silently substitute frames in a frozen plan.

Use actual pad tangent spans as envelope measurements, not guaranteed material
contact areas. Evaluate the central contact sample footprint separately from the
full pad bound. Record surface uncertainty, protrusions, yaw ambiguity and nearby
obstacles. A simulation contact-gap tolerance is not a universal reconstruction
accuracy requirement; noisy observations instead require an explicit uncertainty
model and robustness checks. Empirical p90 disagreement alone is not a calibrated
confidence bound.

Review native segmentation before attributing residuals to shape. Preserve raw
failed scores and mark contaminated masks rather than silently trimming them.
Do not promote incomplete geometry merely because one hypothesis looks right.
A deliberately hypothesis-based mechanics test must remain distinct from
observed-contact admission and hardware readiness. Once prerequisites pass,
follow [control readiness](CONTROL_READINESS.md).

When proceeding with an explicit completed-shape hypothesis, use the separate
[conditional envelope and full-hand checks](CONDITIONAL_HAND_ENVELOPES.md).
Keep that decision distinct from observation admission.
