# Shared exterior-profile fitting with reserved views

[Visible contour constraints](CONTAINER_CONTOUR_CONSTRAINTS.md) are now supported
through an optional `image_constraints` field on each training view. Without this
field, the original endpoint/partial-depth objective described below is preserved.

`real2sim2real.egocentric.container_profile` fits four exterior radii at fixed
fractional heights (0, 0.25, 0.75, 1) and one total height. It jointly fits rigid
poses for the supplied **training views only**. Each view supplies measured
fit-only points, camera intrinsics, uncertain bottom/rim centers and an initial
pose. Reference scans and trajectories are not inputs.

`fit_shared` bounds metric radii/height and applies weak shape priors discouraging
unsupported bulges. Its objective combines visible surface distances and endpoint
reprojection errors. It does **not** constrain observed rim diameter or full
silhouette width. With partial surfaces, pose/depth/radius can trade off; a
radius reaching a bound is an important failure diagnostic. Do not expand bounds
or promote a narrower shape merely because some depth residuals decrease.

Training views can declare `pose_mode`: `free` retains six rigid parameters;
`fixed` retains the supplied pose exactly; `support_translation` fits only two
translations along the supplied local X/Y axes, within 30 mm. Use the latter
only when those axes describe a verified support plane. It preserves the axis
direction and support-normal placement. `endpoint_visible` optionally supplies
two booleans (bottom, rim); a hidden endpoint's placeholder pixels contribute
no image residual. Visible complete contours still constrain the other endpoint
and its image width through `image_constraints`.

`linear_profile` represents the frozen frustum exactly; `distance` computes
distance to the piecewise-linear revolved exterior and external bottom. The mouth
remains open. `surface` samples an open exterior for review, not a completed solid.
The optional `surface_distance` callback in `container_axis.fit_endpoint_pose`
lets both profile representations use the same endpoint/pose-fitting procedure.
Existing callers retain the original frustum distance by default.

Fixed-shape validation can also pass `image_constraints` and a full
`projection_profile` to `container_axis.fit_endpoint_pose`. Both comparison
shapes must use the same adapter and observations. Contour residuals then measure
agreement with fitting cues, while disjoint depth samples provide depth evaluation.

`residual_breakdown` localizes untrimmed errors by bottom band, middle wall, rim
band and outside-height samples. Its labels depend on the input pose/profile and
do not isolate sensor error from geometry error. It is a diagnostic, not a filter
for excluding inconvenient observations.

Freeze the candidate before reading validation outcomes. Fit only rigid poses
on validation views using their fit-only spatial tiles and endpoint cues, then
evaluate the untouched tiles and visible image coverage. Explicitly distinguish
new reserved views from previously inspected regression views and shape-training
frames. Spatial depth holdout shares sensor/calibration errors and is not
independent ground truth. Endpoint errors are fit residuals. One-sided visible
pixel coverage under occlusion can miss oversized projections.

Compare baseline and revised shape on the same observations, pose fitter, limits
and denominators. Reject parameter-bound collapse, regression, or failed reserved
views; keep every candidate and source hash. Shape fitting does not establish
pad support, wall thickness, protrusions or robot insertion. Protrusion exclusions
must be recomputed after an accepted shape/pose change, not copied blindly from
another object coordinate system.

If radius collapse appears, the next measurement is explicit visible rim/bottom
ellipse or silhouette-width evidence, with an occlusion model and calibrated
cross-view checks. Repeated unconstrained profile searches do not resolve a
missing width constraint. Previously used validation views become development
evidence; reserve new views for another candidate.

Tests check exact frustum/profile distance agreement, open-mouth semantics,
synthetic shared-height recovery, invalid input rejection and compatibility with
the endpoint-fitting adapter. They do not establish real-scene accuracy.
