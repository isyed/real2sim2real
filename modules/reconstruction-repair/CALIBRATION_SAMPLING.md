# Calibration sampling and diagnostic tracking

`artifact_metric.prepare(source, output, floor_selection)` retains its default
uniform, twelve-frame sampling. An explicitly recorded alternative is
`sampling_policy="visibility_bins"`. It divides the complete source timeline
into twelve fixed bins and searches nearest each bin's midpoint for an eligible
frame, with deterministic ties. Alternating bins define fit and held-out splits.
Every inspected frame, rejection reason, selected index and threshold is saved
in `sampling.json`, whose hash enters the calibration provenance.

Eligibility uses native object/person masks and the source's own predicted
metric depth only. Defaults require at least 80% mask visibility after excluding
the person, 90% valid depth coverage after two-pixel erosion, 100 valid depth
pixels, no image-boundary clipping, and relative depth IQR at most 0.35. These
rules are declared before scale fitting. They never inspect a mesh, tracked
pose, fitted scale, tracking score or clean trajectory. A bin without eligible
evidence fails; it is not silently dropped or replaced with an easier time bin.

This repairs an evidence-selection failure, not necessarily a reconstruction.
A small exposed fragment can still look internally coherent; projected extent
can vary with viewpoint. The original independent floor, fit/held-out extent,
depth and current-size checks still decide whether calibration passes. Preserve
previous failed experiments, and do not rerun with progressively looser bounds
until a desired size is obtained.

Frame-zero support is always estimated separately at actual source frame zero.
A later first calibration sample must not supply the starting support plane.
A failed frame-zero floor returns no plane and cannot authorize registration.

`tracking_plan.plan(calibration)` remains conservative by default. An explicit
`allow_unchanged_mesh_diagnostic_tracking=True` permits a bounded retracking
diagnosis after a valid, measured initial registration, even when coarse metric
evidence disagrees. That mode forces the unchanged mesh, disables normal fine
repair and returns `reference_promotion_permitted=False`. It does **not** turn
a rejected scale hypothesis into a verified size. Image/depth/support/motion,
thermal and runtime gates still apply, and partial/rejected tracks cannot replace
the source. The downstream caller must retain provenance and require resolved
metric evidence and full independent validation before any reference promotion.

Pure sampling tests exercise deterministic occlusion replacement, missing time
bins, input-grid validation, depth mixing and clipping. Actual checkpoint loading,
GPU execution, source evidence review and downstream physics remain separate.
