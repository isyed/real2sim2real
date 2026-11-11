# Direct stereo checks for uncertain contact surfaces

A projected depth residual mixes stereo matching, resampling into another camera,
object pose, calibration and surface geometry. Before fitting a correction, inspect
the original stereo views. `egocentric.stereo_contact_evidence` provides a CPU sparse
rectified-image matcher with explicit rejection reasons. It searches a declared
disparity interval independently of the model's predicted depth, checks texture,
normalized correlation, competing matches and reverse-match consistency, then
triangulates only accepted correspondences.

The source image location still comes from the reconstruction. This is a
model-seeded consistency check, not independently annotated contact truth. Patch
matching can confuse adjacent surfaces even when correlation is high. A rejected
match is missing evidence, never a zero-error observation or proof of bad geometry.
Subpixel interpolation does not establish subpixel measurement accuracy.

`controller_actor_training.stereo_contact_audit` binds the capture, accepted
calibration and prior contact-region audit by hash. It verifies the original plan,
object geometry, pose timeline and synchronized frame IDs/timestamps. It keeps
camera calibration, object poses and contact geometry fixed. Saved masks in the
target camera and mesh visibility in both stereo views gate contact queries. Mesh
visibility includes outward-normal facing and self-occlusion; it is model-derived
and cannot establish actual surface visibility independently.

The report includes:

- Per-contact fit/held-out match counts and unchanged median/p90 depth checks.
- Missing visibility, low texture, ambiguity and reverse-consistency failures.
- Projected local wall thickness and depth sensitivity to one disparity pixel.
- Textured-image controls, selected as corners in the left image without model
  depth. These check whether matching works elsewhere, not whether contacts are
  metrically accurate. Control successes never count toward contact support.
- RGB and both rectified stereo views, with predicted and observed projections.
- Source and worker hashes. No ground-truth mesh or motion is read.

Wall width is the projected central solid-span thickness along the original
planner's closing axis, translated to each queried contact. It is an approximate
local feature scale, not a measurement of every patch. Depth sensitivity is
`Z/disparity`; it is not an uncertainty estimate. A small patch can preserve a
thin feature yet lose the texture needed for unique matching. Compare explicit
patch-size sensitivities without choosing favorable results post hoc.

```bash
python -m real2sim2real.controller_actor_training.stereo_contact_audit --config stereo.json
python -m real2sim2real.controller_actor_training.stereo_contact_review \
  --audit primary/audit.json --audit small-patch/audit.json --output review/review.html
```

The configuration supplies `capture`, `calibration`, `contact_audit` and each
corresponding `_sha256`; `reconstruction`; a fresh `output`; `deadline_s`; disjoint
nonempty `fit_frames` and `validation_frames` with at most 40 combined capture
indices; and `matching`. Matching fields are `radius`, `minimum_disparity`,
`maximum_disparity`, `vertical_radius`, `minimum_std`, `minimum_correlation`,
`minimum_uniqueness` and `maximum_roundtrip_px`. The worker disables GPU execution
and limits CPU threading. The review must be served from a directory containing
the audit/image directories so relative links remain accessible.

No correction fitter is invoked by this audit. If fit observations are missing,
there is no justified correction to test. If sufficient observations become
available, any proposed correction needs a separate immutable output and held-out
reprojection/geometry evaluation before robot planning. Existing scene, contact,
force and physical-success gates remain in place.

`tests/test_stereo_contact_evidence.py` checks known disparity and metric depth,
low texture, repeated patterns, search-boundary rejection, reverse ambiguity and
bounded input validation. Synthetic matches validate logic, not real grasp success.
