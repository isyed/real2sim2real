# Local contact geometry versus global image accuracy

Admission should depend on the surfaces and free space needed by a proposed
contact strategy. Distant image errors are useful calibration controls, but they
must not independently reject otherwise supported local contact geometry.

Keep these questions distinct:

1. Can repeated physical features be associated across cameras and time?
2. Does native stereo support metric position and local surface orientation?
3. Is the proposed finite pad footprint observed with sufficient coverage and
   uncertainty, and is the complete finger/palm insertion path clear?
4. Can the fitted, actuator-limited hand acquire and retain the free object?

Good sparse registration does not imply good surface coverage. A feature on a
tracking marker or rack edge cannot certify a smooth cup-wall patch. Conversely,
a missing RGB feature does not invalidate a table plane already supported by
stereo. Image-region boxes are selection aids, not semantic mesh segmentation;
background or table may be visible through obstacle openings.

## Reusable diagnostics

`egocentric.multiview_tracks` provides:

- `reciprocal_indices` and `three_camera_identities`: appearance-only mutual
  descriptor cycles with duplicate raster orientations collapsed.
- `stereo_to_target`: triangulation from two native camera rasters, followed by
  projection into a third camera with its distortion. Third-camera observations
  never enter the triangulation. Invalid depths remain unavailable.
- `common_offset_errors`: compare explicit target-frame offsets on one shared
  identity cohort and source-frame set. Invalid observations cannot disappear
  independently for the offset being scored.

`correspondence_audit.image_features(image, mask=None)` accepts a raster mask to
focus detection without cropping or changing intrinsics. Masking also changes
the descriptor search population; record it as a separate condition, not a
silent replacement for whole-image matches. Identity cycles can agree on repeated
texture and remain insufficient without image review and geometric checks.

Retain both unfiltered common-image observations and a separately declared
stereo-quality subset. Explain each rejection, including sample shortage.
Pixel residuals, metric depth and contact clearance must use distinct units and
limits. Short-window tracking does not establish a hardware exposure offset or
rolling-shutter model.

## Scope the next action

Preserve accepted local components such as a supported tabletop plane. Check a
frozen object hypothesis in native stereo before changing its shape to explain
cross-camera projection errors. Distinguish observed surfaces from hypothetical
completion and carry uncertainty into pad placement and collision envelopes.
Do not require complete global calibration before a local geometry assessment.
Do not call a hypothetical-geometry mechanics trial verified reconstruction.

Once local evidence and insertion checks permit, follow
[control readiness](CONTROL_READINESS.md): acquisition, short lift and complete
fixture hold; arm-path audit and arm-mounted hold; then the continuous task.
Agentic proposals and numerical candidates must share those gates.
