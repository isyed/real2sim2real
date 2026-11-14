# Source-derived scene and object motion

Use this path when a moving camera observes an approximately axisymmetric object
and only a local part of its destination must be reconstructed. Keep evaluation
reference poses and scans outside inference. Capture images, calibration,
source timestamps, prior observation-derived shape and world alignment are inputs.

The reusable implementation lives in `real2sim2real.egocentric`:

| Module | Responsibility |
| --- | --- |
| `native_tracks` | Reciprocal native-stereo landmark triangulation and fixed-anchor PnP. Reserved landmark identities never enter pose fitting. Failed poses are NaNs. |
| `planar_stereo` | Calibrated ray/plane intersection and bounded photometric plane hypotheses. Uses a brightness offset, not an image registration warp. Rejects out-of-image projections. |
| `bearing_surface` | Image-cell fusion and a local triangle skin. Image masks, maximum pixel bridges and metric edge limits prevent arbitrary gap filling. Reports connected components and cell depth spread. |
| `axis_motion` | Directed-axis orientation gauge and motion diagnostics on an increasing measured clock. Rejected frames are explicit gaps. |
| `axis_sequence` | Fixed-shape translation/axis fitting against source points and endpoint cues, with timestamp-based acceleration regularization. Outputs hypotheses for separate validation. |

## Camera and object are separate estimates

The first camera pose defines the fixed world anchor. Subsequent poses are
validated by projecting reserved static landmarks into the image. Report that
anchor separately; it has not undergone the same held-out pose test. Image
consistency does not establish independent metric ground truth or object accuracy.

The cup/object frame is bottom-centred, with +Z toward its opening. Estimate
translation and that directed axis. Parallel-transported X/Y axes provide a
continuous coordinate convention but do not measure axial spin. Unobserved yaw
must not drive a protrusion-dependent grasp or be scored as measured orientation.

Endpoint cues can be approximate, occluded or interpolated. Preserve their origin
and pixel uncertainty; fitting residuals against them are not independent labels.
Use reserved surface samples and image constraints for checks. Spatially split
dense stereo pixels are correlated observations, not independent ground truth.

The sequence objective uses a quadratic acceleration prior to discourage large
pose flips. Soft robust losses on acceleration can make a discontinuity cheap
enough to survive. A sequence fit that looks smoother still requires validation:
it can retain depth disagreement, support penetration, missing observations or
an incorrect final placement. An optimizer success flag is not control admission.

## Connected bearing geometry

Reconstruct the visible floor and rim as a finite surface. Preserve disconnected
components and gaps in the evidence export. A selected connected component is
not evidence that discarded components or occluded obstacles are absent.

If a declared floor interior is completed with a plane, export both the observed
skin and the completed candidate. Mark completion-dependent vertices, finite
extent, local interpolation bounds and empirical uncertainty margins. An open
mesh has no measured thickness. Converting it into collision patches requires
an explicit thickness assumption and a new scene version.

Project existing simulated load carriers back into native observations. A small
voxel supporting a large load may disagree with the observed surface. Keep the
old controller as comparison; do not delete unknown obstacles solely to improve
its outcome. Never use a whole-rack convex hull as the physical bearing surface.

## Timing contract and stage handoff

Keep `frame_ids`, absolute `timestamps_s`, and elapsed time obtained by subtracting
only the first capture timestamp. Export accepted poses separately from the full
diagnostic hypothesis; rejected accepted-only entries remain NaNs. CSV exports
that include rejected hypotheses must carry an explicit validity column.

Visual phase boundaries are approximate annotations, not measured contact events.
Robot home approach and return belong outside the source task clock. If execution
needs slower motion, freeze an explicit monotone source/execution mapping rather
than silently stretching the entire trial. A nominal-frame-rate review video
does not replace measured per-frame trajectory timestamps.

Before control, check endpoint support consistency, persistent contact-region
uncertainty and the complete obstacle-restored path. Preserve source gaps in the
report even when constructing an explicitly conditional nominal trajectory.
Do not promote a reconstruction merely because a controller can complete a task
against its assumptions.

See [repeatable reference trials](REPEATABLE_REFERENCE_TRIALS.md) for fixed home
boundaries and [control readiness](CONTROL_READINESS.md) for subsequent gates.

## Placement anchors and physical patch candidates

`placement_anchor.fit_anchor` fits a single static pose across multiple fit views;
reserve complete other views. Optional upright/table contact and endpoint
visibility are explicit assumptions. Report excluded endpoint residuals separately.
A local depth/rim pass can still disagree with the visible body contour. Do not
promote that pose until the disagreement is addressed.

`retargeting.bearing_colliders` converts triangles into individual finite-thickness
convex prisms. It preserves existing geometry and avoids filling the global
concave rack hull. Thickness is assumed, and sharp seams need checking. An additive
scene screen is a first check only: old cell load carriers still exist. Assign
replacement/exclusion roles from geometry evidence before claiming repaired
physical support. Sampled path clearance is not loaded grasp retention.

## Conditional physics comparisons

A permitted nominal experiment can explicitly use uncertain source hypotheses
without presenting them as verified reconstruction. Freeze the source interval,
endpoint alignment, execution time map, unobserved rotation convention and the
same physical stop limits before simulation. Compare scene-only and motion-only
changes separately against an exact baseline repeat.

`controller_actor_training.reference_translation.bounded_translation` retains
baseline orientation and bounds a translation proposal by a gain and metric cap.
Zero gain/blend reproduces the baseline. These bounds are not grasp-retention or
collision guarantees: a kinematically clear proposal can drop the object under
load. Preserve stopped runs and hold their last achieved pose for remaining
score frames; never compress a failure's duration to masquerade as a full trial.

An additive surface can leave saved dynamics exactly unchanged if original
cells still support the object. A real replacement experiment must explicitly
assign local cell roles and measure the new surface's load contribution. Keep
unknown obstacles outside the selected region. Compare task completion, actual
load carriers and pose scores separately; improved geometry need not improve
tracking scores, and tracking metrics need not indicate task completion.

## Phase-limited loaded refinement

`reference_translation.phase_window` provides a smooth, exactly zero envelope
outside a declared interval of the reference clock. Combine it with
`bounded_translation` to change one phase while preserving the target orientation
and acquisition/release targets. Keep the original arm-path knots outside that
interval; resampling or solving IK again can otherwise change an excluded phase
even when its Cartesian correction is zero. Verify the saved command and state
prefix against the frozen baseline, with boundary roundoff reported separately.

Use contact evidence to choose the phase before scoring against evaluation
references. Distinguish loss of one pad from sustained loss of all hand contacts,
and compare object-relative drift with arm tracking error before the drop.
Small arm joint error does not prove stable contact or accurate object motion.
After a complete loaded trial, freeze its outputs and evaluate using the same
declared timing and object registration as the baseline. A phase-limited success
does not establish full-source following or strict control acceptance.

For geometry sensitivity, declare which surfaces move when thickness changes.
Extruding farther below an unchanged contact face tests thickness, not surface
height uncertainty. Measure support on all contact classes, including completed
floors and remaining cells. Removing a supporting cell can transfer its load to
another cell; a successful ablation does not verify the removed space is free.
Report summed contact forces over a stable window rather than selecting one
contact per collider.

`controller_actor_training.contact_quality.contact_quality` summarizes explicitly
selected loaded-phase samples against supplied force, drift and pad-angle limits.
It distinguishes any-pad loss from simultaneous all-pad loss and rejects empty
or nonfinite evidence. Sample fractions are neither time fractions nor success
rates. Preserve task completion, hard collision stops and these contact checks
as separate outcomes. A tangential anchor can overconstrain a rolling contact;
more pressure can worsen load transfer. Neither should replace a passing
controller without complete loaded comparison.

Audit completed floors at actual force-bearing contact points in reserved native
views. Record visibility, depth support and signed optical-depth residuals.
Distance to a sparse observed mesh is a coverage indicator, not automatically
surface error; nearby depth may belong to another surface. A completion-dependent
success remains conditional when those checks disagree, even if its task score
is favorable.
