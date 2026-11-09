# Bounded static-floor depth calibration

`reconstruction_repair.static_floor_sequence` adds a numeric sequence adapter
around the existing `depth_scale.static_floor_gain`. Use it when an independently
reviewed static camera/floor provides a temporal depth anchor. The anchor remains
predicted; this does not establish absolute size accuracy.

`inspect_static_patches` compares at least two reviewed stationary image regions
using a bounded translation search. It rejects low texture, correlation below
0.90 or displacement above two pixels by default. Keep source-image identities
and the sampling policy. This test is evidence on those patches, not a complete
camera-motion solution; rotation, parallax and occlusion still require review.

`fit_sequence` accepts a numeric frame callback returning **raw camera-Z metres**,
native pixel intrinsics, and the human/object exclusion masks. Supply the original
predicted frame-zero floor plane, original depth gain, reviewed floor ROIs and
hashed camera-evidence record. The exact original frame-zero gain is retained.
The existing limits remain: gains within 10% of the original anchor, independent
region agreement within 1%, and floor residual p95 at most 2 cm.

Using already human-aligned depth as if it were raw depth applies the wrong
baseline to the gain bound. Keep raw/aligned encodings and their gain histories
explicit. Adapters must verify input hashes and metre decoding at their boundary.

The sequence result retains every failed frame with a null gain. It never fills
gaps, carries gains forward, drops failed frames or approves a reconstruction.
Individual passing frames can remain useful partial evidence; a complete output
requires every frame to pass. Do not attach changed depth to old object poses
and call it a repaired reference. Revalidate poses, source images, held-out depth,
support and contact, then retarget and perform physics evaluation separately.

The helper changes neither human parameters nor object geometry. Training,
submission promotion and storage/video conventions belong to downstream tools.

`reconstruction_repair.depth_sequence.materialize` can compose an accepted
complete gain recipe with the reconstruction engine's native metric-PNG depth
codec. Optional HDF5/Pillow/engine dependencies load only when called. It verifies
the source hash, timeline, original first gain and 10% gain limits, preserves
original raw payloads and the exact aligned frame-zero payload, and leaves
missing-depth zeros/saturation invalid. Its output receipt explicitly requires
fresh tracking; materializing depth approves no mesh pose or human trajectory.

Floor stability does not imply consistent foreground depth. During handling or
occlusion, predicted object depth can disagree with the source silhouette even
when every floor gain passes. A coarse diameter-based size hypothesis may also
produce a complete, smooth track with worse image agreement. Compare current-size
and resized hypotheses on matched source frames, and declare initial resting
contact from source evidence separately from later lifting phases. Keep useful
depth/motion components without splicing mismatched meshes, poses or missing
frames into an approved reference.
