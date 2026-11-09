# Reconstruction repair

For a complete but discontinuous observed object track,
`pose_regularization.propose` builds a bounded smooth prior in camera-space
object centres and local rotation corrections, preserving explicit anchors.
It does not validate source observations or fill absent frames.
`observed_pose_graph.fit` offers sparse joint fitting to visible outline extrema,
explicit first-surface depth correspondences, predicted support and temporal
acceleration. Both are proposal tools: independently remeasure the original
triangles and source masks/depth at every frame, check temporal joins and count
large rotations as well as percentiles before native GPU validation. Fixed depth
correspondences and convex outline proxies can trade off against source image
agreement; an improved fitting cost does not authorize promotion.

For a complete observed sequence that fails a native boundary check by a small
amount, `native_translation_polish.propose` offers a bounded, fixed-rotation
translation search. It uses the exact native triangle silhouette and predicted
surface depth, keeps both real adjacent poses within the original8cm limit,
and never fills a missing frame. The caller supplies its minimum silhouette
agreement, a small metric displacement bound and a finite time budget. The
returned result is only a CPU proposal: preserve source/pose/fitter hashes,
independently validate with the actual GPU renderer, recheck source appearance
and all changed boundaries, then rerun any affected retargeting/physics. Sparse
vertex-depth agreement alone may miss a native triangle-depth boundary failure.
The converse can also occur: a dense triangle-depth median can pass while the
native tracker's sparse front-most vertex-depth median fails. The opt-in
`require_sparse_native_depth=True` checks both contracts, using the native
float32 pose boundary, with an explicitly supplied support plane. It leaves
rotation entries unchanged and records both depth measurements. Default dense
validation remains unchanged. `silhouette_scope='human_excluded'` is explicit;
it must match the caller's native visibility gate rather than relabel a failed
full-silhouette test.

For an explicitly reviewed resting phase,
`support_surface.project_below_support` proposes a bounded change to only
below-plane mesh vertices along their camera rays. It preserves the initial
projection and poses, while reporting geometry/volume/search limits separately
from source and physical checks. This nonrigid hypothesis requires mass-property
recomputation and held-out RGB/depth plus settling validation; it does not infer
correct hidden geometry. See [support-surface proposals](../../docs/SUPPORT_SURFACE.md).
The opt-in `projection_mode='support_normal'` minimizes vertex displacement
instead of preserving the initial projection; it requires the same held-out
source checks. Camera-ray projection remains the default.

For a visibly resting upright prop, `resting_pose.propose` can generate a bounded
pose hypothesis from an explicitly reviewed object axis, support plane and
pivot. It preserves mesh geometry and inertia. Source image/depth preservation,
transition continuity and CPU/GPU settling checks must still qualify the result;
the helper does not infer that every object should be upright.

The complete portable repair engine, algorithm version 0.11.0, is included.
It has no dependency on a neighboring workspace or cloud storage. The pipeline
adapter consumes the common reconstruction artifact; the low-level CLI also
accepts the engine's native bundles:

```bash
python -m real2sim2real.reconstruction_repair.cli --help
```

For monocular object-depth outliers, `object_depth_spikes.plan_masked_depth_spikes`
proposes bounded, temporally bracketed corrections from the object's masked
depth median and screen position. It changes no source data itself. A caller
must preserve the original depth, record every changed frame and input hash,
then independently validate the resulting pose sequence and physics. Large or
unbracketed intervals remain uncertain.

The normal gate order is metric-size calibration first, bounded fine object
pose/continuity repair next, and optional human repair last. Size evidence is
retained separately from later tracking decisions. A proposed resize requires
retracking and revalidation; a better silhouette alone does not certify depth.

Each run records accepted, rejected, disabled and unavailable stages, reasons,
per-frame diagnostic values and the accepted output bundle. Parent hypotheses
are preserved. The engine includes silhouette-preserving temporal correction,
RGB pose hypotheses, rotation/translation continuity checks, support placement
diagnostics, and native-MHR human fitting with explicit evidence. Not every
capability is enabled for every input.

See [the evidence contract](../../docs/DATA_CONTRACTS.md). The orchestrator
requires an explicit `metric_evidence` NPZ and reviewed `floor_roi_source`;
it does not invent a floor plane to force a pass. Human fitting additionally
needs a hashed `mhr_decoder` specification, appropriate human observations,
`human_metadata` (native-to-world transform, decoder identity and parity
provenance), and independently exported `human_parity` geometry. These are
checked before fitting; the adapter cannot manufacture its own evidence of
decoder correctness.
Its decoder is CPU-only. Set `execution.cpu_only: true`.

The accepted output is a reconstruction candidate, not an actor-training
approval. Reference contact, physical contact, object stability and task
completion must be evaluated separately.

`interaction_translation.propose` is an opt-in translation-only temporal fit
for jitter near reconstructed hands or other observed landmarks. It keeps
rotations and explicitly locked frames unchanged, and constrains frame steps,
total displacement and landmark-to-surface separation during optimization.
`surface_witnesses` obtains actual nearest mesh points; bounding their distance
after translation conservatively bounds the distance to the mesh. These are
unsigned geometric distances, not inferred grasp/contact labels. Use paired
landmarks in the same metric camera frame, preserve the parent, and independently
check exact final surface distances and native image/depth/support consistency.
A feasible optimizer result is not acceptance; image preservation can still fail.
No constraints or existing pipeline defaults are relaxed by this helper.

`pose_candidate.prepare` packages a complete independently proposed object pose
trajectory while preserving the parent's other reconstruction files. It verifies
the source hash, full timeline, proper rotations and positive scale, and labels
the inherited quality report as parent-only evidence. Package acceptance is not
tracking or physics approval. `elongated_shape.cross_section_audit` compares
visible mask breadth, predicted depth and mesh breadth to flag a possibly
missing transverse component. It changes no pose or mesh. A flagged shape
should be reviewed on RGB and rebuilt from a suitable visible source frame
before attempting to fix tracking by smoothing or scaling the old mesh.

For an independently calibrated mesh whose fresh seed floats above a reviewed
resting surface, `support_seed.propose_resting_seed` offers a bounded CPU
plane-normal search. The caller must explicitly identify the frame's resting
phase; the helper does not infer or propagate contact. It selects the smallest
sampled translation passing the actual parent's image, depth and support checks,
returns the original seed unchanged on rejection, and records every attempted
move. Default maximum translation is 5 cm. This is a seed hypothesis requiring
fresh tracking, not a trajectory repair or permission to train a controller.

After selection, `real2sim2real.comparison_video.render` can show source RGB
beside complete before/after object and human surfaces. Decode saved humans with
`human_surface.decode_surface` first. The CPU renderer preserves all source
frames, checks matching timelines, verifies the encoded video by decoding it,
and caches only matching video/geometry/surface identities. Its transparent
silhouettes do not establish occlusion, grasp or physical correctness.

A **rejected calibration** is distinct from a validated resize hypothesis. Its
report records `inconsistent_extent_frame_ids` and per-frame relative scale
disagreement. Keep the input mesh and inspect those native RGB/depth/mask views;
do not apply the displayed scale multiplier merely because it was calculated.
Normal fine repair remains blocked until current-size evidence is consistent.
The runner recommends resize/retracking only when an independently retained
calibration hypothesis passes its evidence checks and requires a different size.

### Bind exported poses to their source (CPU only)

For a source-only check of an already exported reference, use
`source_pose_audit.corresponding_mesh_transform` with explicitly corresponding
vertices, then `bind_pose_frames` with the complete source/export pose sequences.
A mesh-local rotation can be large while being correctly cancelled by the pose
conversion. Check the combined transforms before diagnosing an axis error.
Vertex correspondence and triangle identity must be established by the caller;
the fit is not a general mesh registration method.

`source_pose_audit.projection_audit` rasterizes the complete triangle mesh on a
native-grid region containing both the projected mesh and the object mask. It
returns raw silhouette overlap, separately labelled human-mask-excluded overlap,
and consistency with the supplied predicted depth. There is no resizing,
decimation, tracker inference or candidate approval. A near-symmetric object can
keep good silhouette/depth agreement while its yaw is wrong: review distinctive
RGB features and temporal anchors too. A failed depth comparison does not prove
the pose is wrong, because the monocular depth prediction can be unreliable.
Keep diagnostic pose hypotheses separate from accepted artifacts and validate
any proposed repair independently before physical execution.

`appearance_yaw` offers opt-in CPU proposal tools for a fixed coloured mesh:
centroid-preserving yaw, visible-vertex RGB costs, bounded circular-state paths,
and explicit source-reviewed landmark reprojection. Neither lower aggregate RGB
loss nor smoother rotations establishes correct orientation. Broad white surfaces
can dominate the loss while a distinctive panel points the wrong way. Require
semantic landmarks, a front-facing check and separately checked source frames.
The landmark check alone does not establish full-mesh visibility.

Its optional centroid-translation proposal has explicit plane semantics:
`gravity` preserves height; `camera_depth` preserves optical-axis Z and makes
no height claim. Both retain a finite metric displacement bound. Recheck native
triangle silhouettes, predicted depth, support and both ends of a repaired
interval. A good local repair can still create a large discontinuity where it
joins an unrepaired tail. Uniform size changes are separate hypotheses requiring
fit/held-out evidence, metric qualification and new physical support checks.
`project_vertex_colors` is a labelled dense-point preview, not the triangle
acceptance renderer or evidence of physical execution.

The separate opt-in surface_appearance.bake helper can copy source RGB to
front-facing vertices using the full triangle z-buffer and explicit object/human
masks. It keeps geometry and poses unchanged, records observed source frames,
and never paints hidden back surfaces or known human occluders. Such appearance
refinement needs independently checked tracking frames; it is not required for
the vertex-colour loader fix and must not be mistaken for pose validation.

### Qualify contradictory object depth (experimental, opt-in)

For a CPU audit of pose-versus-depth disagreement, use
`surface_depth.front_surface_depth(vertices_camera, faces, intrinsics, image_shape)`.
It rasterizes native triangles with perspective-correct camera-Z interpolation
and a first-surface z-buffer. Sparse vertex projection can sample rear surfaces
or leave gaps; this diagnostic avoids that sampling dependency. It returns
metres and infinity for uncovered pixels. Exclude observed occluders and invalid
depth before comparing, and record the pixel convention (integer pixel centres),
source hashes and any mask erosion. Near-plane crossings fail explicitly.
This opt-in diagnostic does not change existing tracker acceptance thresholds or
turn agreement with monocular predicted depth into ground-truth validation.
`surface_depth.compare_surface_depth` applies a boolean object mask, optional
occluder exclusion and explicitly requested edge erosion. It rejects insufficient
valid pixels and reports predicted-depth consistency separately from metric truth
or candidate approval (both remain false). Full/crop predictions from one model
are sensitivity checks, not independent validation. Do not combine each frame's
most favorable depth source and present that selection as held-out evidence.

`depth_evidence.qualify_depth` can mark unsupported predicted object depths as
missing before calibration or fresh tracking. It requires explicit RGB-reviewed,
bracketing temporal depth anchors for the same rigid object, compatible metre
depth/native boolean masks, substantial foreground support, and consistent anchor
depths. Frame IDs assume uniform temporal sampling. The caller must preserve the
source hashes, anchor selection and per-frame receipts in a derived artifact.

The helper uses no object poses, mesh size, or calibration fit/held-out labels.
It changes only visible object pixels outside the occluder mask, and only when a
substantial contradictory depth population coexists with a consistent foreground
population. Those pixels become zero/missing; no surface depth is imputed. It
returns an unchanged copy if support is insufficient or the frame is consistent.
Keep the original depth and mark the derived artifact as requiring revalidation.
This is not a default reconstruction filter or permission to train an actor.
After qualification, rerun the same retained calibration samples and native
image/depth/support checks; a passed depth-evidence check does not accept a resize
or its full pose trajectory.

## Local translation repair without rotation drift

`TemporalPoseConfig(preserve_rotations=True)` fits translation corrections while
preserving every source rotation exactly, including non-knot and held-out frames.
The default remains joint translation/rotation fitting. This option is useful
when image-supported rotations are already steady and the remaining defect is
camera-depth/translation jitter. It does not turn smoothing into evidence of
correct motion: the existing mask, previous-good-image, depth, support, anchor,
displacement and motion-benefit gates still apply. The current mesh must pass
metric-size preflight before normal window refinement.

After independently accepting bounded windows, use
`localized_pose.merge_windows(parent, frame_ids, windows,
preserve_rotations=True)`. It requires exact locked endpoint poses, ordered
nonoverlapping original frame IDs and accepted window decisions. It preserves
all untouched poses and checks continuity on the **complete** merged clip,
including its seams. A rejected window, changed anchor, changed locked rotation
or remaining motion violation stops assembly. No interpolation or coordinate
transform is inserted by the merge. A numeric merge pass is not full reference,
interaction, physics or submission approval; callers must retain image/depth
receipts and independently validate contact/support and downstream execution.

`localized_pose.backtrack_translation` can reduce an overly aggressive
rotation-preserving proposal using a declared decreasing list of fractions. It
recomputes the same image/depth/support/motion and displacement gates for every
trial. An optional boolean editable-frame vector limits the correction to
explicitly nominated frames; all other poses and locked anchors remain exact.
An optional interaction callback may veto a geometric pass, and can never
approve a geometric failure. Failed trials return the original window. Retain
all trial measurements and describe this as development selection, not an
unseen held-out evaluation. No acceptance threshold is weakened by backtracking.

## Tracking-first bounded repair

`tracking_plan.plan` keeps size calibration separate from pose tracking. A valid
resize hypothesis requires a newly generated tracking input and a fresh
registration; it never rescales the old pose sequence. Inconsistent calibration
retains the original mesh and permits a bounded registration diagnosis only.
Normal fine repair remains blocked. `tracking_plan.assess_replacement` requires
the complete accepted-state trace, full-clip motion checks, unchanged native
image/depth/support gates, current-size evidence and a measured image improvement.
A better mean cannot hide regressions on previously good frames. These decisions
do not approve actor training or physics. GPU execution, resource deadlines and
scenario-specific support evidence belong to the caller.
# Fresh-registration pose refinement

`local_registration.propose_initial` fits a single unaccepted registration
hypothesis against visible object masks, predicted depth and explicit support
evidence. It has no previous accepted temporal pose. Search remains bounded to
8 cm / 30 degrees around the supplied hypothesis; native silhouette, depth and
support checks remain unchanged. It is a CPU proposal and requires independent
GPU revalidation before tracking. It does not approve a hidden trajectory,
continuity across an occlusion, or a complete reconstructed reference.

For bounded temporal recovery fits, `LocalRegistrationConfig.depth_violation_weight`
can prioritize resolving depth errors in the optimization objective. The default
is zero, preserving existing proposal behavior. This does not alter image, depth,
support or motion acceptance limits. The tracking adapter exposes this option as
`--continuous-depth-weight` with `--continuous-cpu-recovery`; both CPU proposal
diagnostics and actual GPU revalidation remain required.

`tracking_windows.refine` composes proposal tracking with anchored multi-frame
checks. It preserves explicit forward/backward source frame mappings and the
recorded source frame rate. Approximately 31-frame windows use alternating
fit/held-out frames. Already-valid proposals stay exact; otherwise the existing
temporal pose fitter must pass all window gates. Rejected/unprocessed frames
remain unaccepted, and a failed window stops that direction. Source hashes,
window decisions and an acceptance ledger are retained. This diagnostic workflow
does not certify metric size, hidden motion, full-clip joins, or physical execution.

# Initial resting phases

For placement-height failures and ambiguous box orientations, see
[tracking through pickup and placement](../../docs/PLACEMENT_TRACKING.md).
The bounded `silhouette_translation` initializer, `anchor_path` hypothesis
selector and rotation-preserving local-registration option keep proposal
generation separate from unchanged native acceptance gates. Videos can carry an
explicit uncertain-frame ledger; none of these options approves a reference.

# Verified tracking continuation

`tracking_resume.verified_prefix` reads a hash-bound seed and tracker state
ledger. It preserves only consecutive rigid accepted poses and rejects a broken
previous-pose chain or accepted rows after failure. Supply `source_frame_ids`
for a cropped or reverse-time window: tracker-local indices and native source
frames are different coordinate systems. A partial prefix never fills the tail
and never approves a whole reference. Revalidate a deliberately revised endpoint
against its source image/depth and the preceding pose before resuming from it.

`lookback_recovery.propose_pair` offers a bounded two-frame proposal when a
recent accepted endpoint prevents the next update from passing depth and motion
checks. Supply the preceding trusted pose, both native images/masks/depths and
explicit consecutive source IDs. Each proposed camera-ray adjustment must pass
the original anchor image, depth, support and temporal gates; the subsequent
local fit must independently pass the same limits. Earlier poses stay exact.
The helper returns no pair on failure and never fills an unverified tail.
CPU success still requires actual GPU-renderer revalidation before tracking.
The receipt names **both** `gpu_revalidation_source_frame_ids`. If ordinary
tracking from the revised anchor does not find the qualified next pose, that
second CPU proposal may be tested directly against its native GPU-rendered
image/depth/support evidence and its motion join. Resume from the second pose
only after both GPU checks and both joins pass; validating the anchor alone
does not validate the second pose. Never treat CPU approval as accepted motion.

Tracking-window receipts expose `tracker_update_count`: the number of frames
after the initial registration/seed, rather than the total window length. Bound
`--recovery-frame-budget` by that count and check `configured_limits` on CPU
before launching tracking. A one-frame window supports registration only.

`crop_depth_sequence` regenerates a declared native RGB window using a fixed
camera-centered crop, reviewed static-floor scale and held-out floor pixels.
Full-image controls are predeclared diagnostics, not pose-conditioned selectors.
Its metric-depth composite preserves pixels outside the crop and leaves invalid
new pixels unobserved. Original depth remains immutable. Every pose must be
revalidated against this new predicted depth before a repaired reference is used.
The archive explicitly describes its raw/aligned arrays as a derived composite;
it does not pretend to be the original model output or independent metric truth.
The sequence finalizer restores outside-crop pixels from the original uint16
values after encoding. Copying float metres alone cannot guarantee this: a
legacy truncating encoder can otherwise lose one millimetre on round trip.

Before a full sequence, `foreground_depth.qualify_for_pose_validation` can
screen an explicitly declared set of full-image/context-crop samples. It checks
the exact native sample IDs, held-out floor decisions, visible foreground
coverage and depth changes between adjacent native samples. A good floor fit
alone does not establish valid depth on a small object. Sparse gaps are reported
as untested, and a passing result only permits independent native pose checks;
it never accepts a repaired motion reference or claims measured metric truth.
Use a fixed camera-centered crop with its adjusted FOV/intrinsics, and preserve
all rejected inputs and variants rather than silently selecting by desired pose.

`foreground_visibility.audit` accepts complete contiguous native windows with
nonzero starting frame IDs. Warning, occlusion-gap, preceding-observation and
reacquisition IDs retain those native indices. Raw/aligned comparisons require
matching native IDs, not merely equal frame counts. A window audit is still an
input diagnostic; a smaller number of depth warnings does not qualify a pose.

For before/after videos, `real2sim2real.comparison_video.render` accepts explicit
`available_frame_ids` with `allow_partial=True`. This supports a qualified
prefix plus a later independently reacquired interval on one source timeline.
Frames between them show source RGB and a missing-pose notice; neither the
object nor human overlay is carried through a gap. Receipts distinguish array
coverage from actually available poses. This is diagnostic media, not approval
to splice disjoint intervals into a physical motion reference.

After anchored CPU fitting, `real2sim2real.reconstruction.validate_pose_sequence`
checks explicit native frame mappings, every proposed pose and its temporal join
with the actual GPU renderer. It stops at the first rejection and never fits or
carries a pose. Passing an interval does not approve whole-reference seams,
physical execution, training or deployment.

`observation_preservation.preserve` is an optional post-fit proposal step. It
restores an original **fit-frame** pose only when that pose passes all native
image/depth/support/camera checks and the temporal fit regressed its image
agreement. Held-out frames are never chosen for restoration. The entire hybrid
window, anchors, displacement and motion bounds are then checked again; GPU
validation remains required. Inspect the reported acceleration as well as step
bounds before physical use, since preserving an observation can alter smoothness.

When endpoint interpolation misses the observed path, `anchored_recovery.refine`
also accepts independently checked interior anchors. A bounded follow-up may
register one near-midpoint **fit** frame and refit with all three poses locked.
Keep held-out frames out of anchor selection, retain the rejected two-anchor
trial, and recheck the full interval with the original limits. Extra observed
anchors are preferable to silently increasing the permitted rotation change.

`dense_anchored_recovery.refine` can warm-start an interval using additional
pose hypotheses from declared fit frames. Held-out initialization poses are
forbidden. The original anchor-only path remains the acceptance baseline;
proposal poses are editable, and registered anchors remain exact. This addresses
initialization only and does not relax any depth, image or motion limits. See
[the depth-window workflow](../../docs/DEPTH_WINDOW_REPAIR.md).

# Resting-support proposals

For slender-object rotation failures, `elongated_shape.principal_axis` and
`orientation_audit` compare the proposed projected mesh axis and image position
with visible native mask pixels. End-on/insufficiently visible views are marked
uninformative; axis sign and roll about the shaft are not observable from this
comparison. It diagnoses tracking drift without changing a pose, replacing
native acceptance gates or certifying metric size.

For explicit complete candidates assembled from measured intervals,
`pose_candidate.compose` preserves native IDs and records source ownership,
failed gate overrides and all inter-frame joins. `pose_candidate.prepare`
packages the unchanged paired human/mesh/camera while retaining parent quality
separately. See [the interval handoff workflow](../../docs/POSE_INTERVAL_HANDOFF.md).

`support_seed.resting_orientation_hypotheses` enumerates nearby convex support
faces whose projected centre lies inside the face. It preserves the world centre
and leaves contact translation to a separately checked proposal. Declare source
evidence that this is a resting phase; geometric support is not a physics proof.

`support_seed.local_rest_orientation` is an optional, evidence-declared
initial-rest hypothesis. It rotates about a mesh-local centre, caps the angle,
uses explicit hold/fade source-frame IDs, and preserves every later pose exactly.
It does not accept its own output. Check images, predicted depth, seams,
retargeting and actual physics before adopting the result. See
[the static-support and initial-rest workflow](../../docs/STATIC_SUPPORT_SETTLING.md).


### Assembly of measured pose intervals

`pose_segments.merge` combines independently validated forward/backward pose segments while preserving previously accepted baseline poses exactly. Unavailable-frame masks remain explicit. Internal motion and newly introduced joins are checked; failing new boundary frames are removed rather than interpolated. Callers retain and verify actual validation receipts. A sparse diagnostic does not authorize retargeting or physical execution.


### Resting-phase scale/depth ambiguity

`resting_scale.propose` generates an explicitly reviewed ground-supported camera-ray scale hypothesis with fit/held-out checks. Mesh and camera translations change together, preserving pinhole projections; native depth and interaction acceptance remain separate. See [the contract and limitations](../../docs/RESTING_SCALE.md).

`fixed_scale_pose.restore_fixed_scale_projection` handles a related export
contract: when a monocular fitter returns a changed uniform object scale,
restoring a calibrated mesh size requires scaling its camera translation by
the same ratio. Overwriting scale alone changes the image footprint. The
helper rejects anisotropic scale and preserves pinhole projection; it does not
establish correct metric depth, contact or temporal continuity. Audit the final
exported pose after all scale overrides, not only the fitter's intermediate IoU.
# Joint depth and pose hypotheses

`joint_depth_pose.fit` re-estimates uncertain pose anchors and latent foreground
depth using multiple named source predictions with a fixed camera and mesh.
`refine` refreshes fitting-frame surface correspondences under a shared budget.
Only fitting observations enter the optimizer; validation must use unchanged
depth estimates and withheld frames. These functions never approve a reference.
See [external recovery and depth disagreement](../../docs/EXTERNAL_OBJECT_RECOVERY.md).
