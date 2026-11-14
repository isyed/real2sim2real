# Tapered-container hypotheses

[Shared profile fitting](CONTAINER_PROFILE_FITTING.md) provides a separate
four-radius/height hypothesis with training/validation separation. It preserves
the frozen frustum and exposes parameter-bound collapse and regressions.

[Endpoint constraints and protrusion exclusions](CONTAINER_AXIS_AND_EXCLUSIONS.md)
add uncertain rim/bottom cues, fixed-world directed-axis diagnostics and explicit
keep-out hypotheses while retaining frozen dimensions and validation limits.

`real2sim2real.egocentric.tapered_container` fits an explicitly assumed
axisymmetric outer envelope from measured wall points and an image outline.
It does not replace the observed scene or estimate wall thickness. The resting
fit assumes the bottom lies on the level support and the axis is upright.

`fit_resting` fits center, base radius, top radius and height, with bounded robust
least squares. `surface` samples the exterior and its rim; it does not complete
the interior. `outer_distance` measures distance to the tapered side or external
bottom, leaving the mouth open. `fit_pose` keeps dimensions frozen and fits a
rigid pose from observed points using multiple axis initializations.

For later views, select visible object pixels from images, erode boundary pixels
and split reliable stereo observations into disjoint spatial tiles. Fit pose
using one set; score the other. Report both median and tail errors. These sets
share sensor and calibration errors and are not independent ground truth.
An observed-pixel-to-projected-envelope check allows hand occlusion but is
one-sided: an oversized envelope can pass it. It is not silhouette IoU.

Inspect overlays and record failed poses. Axisymmetric yaw is unobservable;
end-for-end ambiguity and partial-view fitting can also produce incorrect poses.
A failed view does not isolate geometry from registration error. Explicit rim
and bottom observations, multiview consistency, and visibility-aware rendering
are needed before promoting a hypothesis. Small protrusions, rim lips, inner
geometry and material thickness require separate treatment.

Keep fitting and validation frames separate, pin source and implementation
hashes, freeze dimensions before later-view pose fitting, and save rejected
results. A partial validation pass is not contact admission or permission to
skip insertion, reachability and stationary-hold gates.
