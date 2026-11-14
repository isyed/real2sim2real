# Correspondence identity before camera-model fitting

A large calibrated epipolar residual is not automatically a distortion error or
an object-depth error. Keep the original images, calibration and geometry fixed
while separating spatial concentration, correspondence identity and image motion.

`egocentric.camera_orientation.pair_residuals` exposes signed Sampson distance,
cheirality and source depth in the original match order. `pair_geometry` retains
its original aggregate interface. Sampson distance is measured in undistorted
pixel coordinates; it is not full 2D projection error or millimetres of depth.
Its sign follows the directed camera pair and must not be compared across pairs
as if both used the same orientation.

`egocentric.correspondence_audit` provides:

- `image_features` and `reciprocal_support`: raw SIFT identities and strict
  bidirectional descriptor ratio support for saved matches.
- `cycle_consistency`: join A→B→C to A→C by pixel identity. Missing and ambiguous
  joins are unavailable (NaN), never successful zero-error observations.
- `spatial_groups`: raw image grid and principal-point radius normalized by half
  the raster diagonal; no fitted correction.
- `error_summary`: signed median, absolute median and p90; empty cohorts remain
  explicitly unavailable.

Preserve all original matches. Report identity-supported subsets alongside the
full set, rather than removing large residuals and reporting the smaller score
as calibrated accuracy. Keep match IDs, source hashes and per-point arrays for
review. Duplicate feature orientations at one pixel are correlated evidence.
Three-camera cycles and reciprocal descriptors can also agree on repeated
texture; neither replaces independent landmark review.

Render spatial maps and paired raw-image crops with IDs. Compare recognized
static landmarks across views as well as bins: image locations, depth, visibility
and camera motion all change with viewpoint. Track adjacent-frame motion as a
confound check, not an exposure timestamp measurement. A time-varying error field
does not by itself establish rolling shutter or a stream offset.

If camera parameters are subsequently fitted, declare fit/validation views and
bounds first, keep stereo and target-camera changes distinct, and retain the
original calibration. Validate the local contact region, not only distant scene
features. A global background p90 is not a grasp-admission metric.

For grasp development, retain the [control readiness sequence](CONTROL_READINESS.md).
Deterministic/numerical planning and an agentic proposal generator can share the
same scene, hand fitter, collision checks, force limits and physics evaluator.
Neither planner admits unknown contact geometry by producing a plausible pose.
