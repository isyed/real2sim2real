# Trace stereo samples into a target image

`real2sim2real.egocentric.projection_trace.project_with_source` projects an
organized metric point grid into a target pinhole image with the same float32
z-buffer convention as `camera.project_depth`. It returns the target depth,
winning source-pixel index and unrounded target pixel. Empty target pixels retain
source index −1. Depth collisions select the nearest observed source, with a
deterministic source-index tie break.

This allows a consumer to replay a frozen target depth and follow each selected
residual back to the actual stereo pixels. Compare the original source point
transformed into the target camera with target-depth unprojection to quantify
pixel-rounding effects. Exact replay verifies software provenance; it does not
verify physical visibility, calibration or synchronization. Target-camera-only
occluders may be absent from the source depth.

For diagnostics, compare error-conditioned samples with low-error controls and
declare the selection and full disparity/depth search bounds. Use the existing
`stereo_contact_evidence.match_rectified` for ambiguous/reverse-match checks,
and `multiview_points.reciprocal_match` for cross-camera appearance checks. Both
use the same input images and are not independent ground truth. Report rejected
matches explicitly; a correct smooth patch can be rejected, and repetitive
markers can still produce a wrong accepted correspondence.

Preserve raw high-error observations, geometry and calibration throughout the
audit. Large source-to-target appearance offsets may involve matching error,
extrinsics, time alignment, rolling shutter, occlusion or model/pose mismatch.
Identical encoded PTS alone does not establish hardware synchronization. Do not
calibrate against the reconstructed surface merely to force agreement.

Tests verify nearest-surface provenance, missing/invalid points and exact agreement
with metric projection under rigid translation and pixel rounding.
