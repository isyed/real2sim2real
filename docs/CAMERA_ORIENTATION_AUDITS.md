# Raster orientation and optical calibration audits

`real2sim2real.egocentric.camera_orientation` constructs explicit 180° convention
hypotheses without changing video pixels or existing reconstruction artifacts.
Do not treat an acquisition `rotation_deg` field as an instruction to transform
calibration again: the released model may already describe the delivered images.

For a literal raster half-turn with a simultaneous optical-axis change
`S=diag(-1,-1,1)`, positive-focal OpenCV intrinsics obey:

```
cx_new = width - 1 - cx
cy_new = height - 1 - cy
```

Focal lengths are unchanged. Radial coefficients are unchanged; tangential,
thin-prism and tilt coefficients reverse sign. `turn_pixel_calibration` supports
4/5/8/12/14-term OpenCV perspective distortion. Tests use `cv2.projectPoints` to
verify exact agreement with a literal pixel rotation, including nonzero tilt.

Extrinsic axes are a separate hypothesis. When both source and target camera
axes change, use `T_new=S4 @ T @ S4`, with the homogeneous four-dimensional S4.
`convention_streams` independently toggles pixel calibration and optical axes,
allowing audits of already-processed or partially processed metadata. These are
candidate conventions, not automatic repairs. Preserve stereo order and baseline
length and inspect translation direction explicitly.

`pair_geometry` evaluates fixed raw-pixel matches after distortion correction:
calibrated Sampson error and positive depth in both cameras. No pose is fitted.
Translation reversal can leave epipolar error unchanged while placing all points
behind the cameras; the tests exercise that failure. Sampson residuals differ in
definition from rectified vertical disparity and should be labeled accordingly.

Use identical correspondences for every hypothesis, declare discovery/held-out
frames, and keep RGB matches on static regions when exposure timing is uncertain.
Inspect decoded raster dimensions and MP4 rotation/display metadata separately.
Metadata absence cannot reconstruct an undocumented publisher export pipeline.

Evaluate any previously accepted stereo refinement separately from the raw
convention comparison. A convention may be clearly preferable while still
requiring calibration refinement; low stereo error alone does not establish RGB
registration, accurate depth or control readiness. Save rejected manifests and
never overwrite the active capture merely because a rotation flag exists.
