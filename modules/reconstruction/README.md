# Reconstruction

This module contains the existing RGB segmentation, human initialization,
metric depth, object generation and pose-tracking implementations. The new
stage adapter connects them in this order:

1. SAM2 segments the person and manipulated object from explicit prompts.
2. MoGe2 predicts depth and camera intrinsics; SAM 3D Body estimates native
   MHR motion. Depth is aligned to that predicted human geometry.
3. The aligned native-resolution depth is unprojected with the same camera
   into a metric point map for SAM 3D Objects.
4. SAM 3D Objects generates the mesh; its predicted scale is applied once.
5. Ordinary FoundationPose tracks that mesh using the same aligned depth
   and camera, with bounded registration/recovery and quality diagnostics.

The important integration correction is **shared metric input before mesh
generation**, rather than generating a mesh in a separate depth frame and
trying to fix all size discrepancies afterward. This reduces an avoidable
inconsistency; it does not make monocular metric predictions ground truth.
The default stage performs initialization, not a full joint refinement.

### Optional external tracking recovery

`hybrid_tracking.run_sequence` runs a supplied tracker normally and calls an
external pose provider only when its update fails independent checks. This lets
an expensive method propose occasional recoveries without replacing good normal
tracking. The provider is model-independent; its own quality score is not trusted.

Each proposal must retain the same rigid mesh, metric units and common camera
frame. The measurement callback checks visible silhouette, predicted depth and
support penetration. Recovery uses the existing stricter image threshold. Motion
limits use actual elapsed source time and permit only an explicitly bounded gap;
rejected poses never become the next tracking seed. Missing observations stay
NaN with a separate acceptance mask. This sparse diagnostic output is **not** a
complete robot reference or an automatic reconstruction promotion.

The optional `hybrid_tracking_native` CLI compares FoundationPose alone with
FoundationPose plus a hash-bound external proposal cache. It validates source
IDs, camera, mesh and offline model inputs before GPU work. Cached proposals
from another tracking chain are a replay ablation, not a measurement of an
integrated on-demand system. A positive short test should precede a longer
state-conditioned recovery test. Dataset selection and thermal/session guards
belong to the downstream runner. This adapter is opt-in; the default pipeline
does not change.

Retain both raw and image-refined external poses when available: image refinement
can improve overlap while worsening metric depth or motion. A rejected pose must
not be rescued solely by its overlap score. When a proposal narrowly misses a
motion bound, also inspect its depth and support residuals before attributing
the failure to that bound. Disagreement with monocular depth may implicate the
depth estimate, the pose, or both; record the uncertainty rather than silently
relaxing validation or altering the source evidence.

`conditioned_recovery.select` adds a bounded selection step for competing depth
hypotheses. It requires every declared hypothesis to pass independently; it
cannot choose depth according to which one makes a pose pass. Keep every sampled
pose for this check, rather than exporting only the sampler's image-ranked
winner. `conditioned_recovery_native` supports explicit per-frame file handoffs
so the proposal model and native tracker can use a small GPU sequentially. It
binds each proposal file to the previous accepted pose, source frame, camera,
mesh and file hash. This is an opt-in experiment, not default tracking behavior.
The repair module's `canonical_pose` converts that accepted metric pose into and
out of fixed mesh normalization without changing the mesh's physical size.

For a long or partly occluded object, set `options.mesh_frame_index` to a
reviewed source-video frame where the complete shape is visible. The default is
frame zero. `metric_inputs` decodes that exact RGB frame and pairs it with the
same-index propagated object mask and aligned metric depth; `metric_mesh` checks
the frame receipt before inference. The receipt pins source-video, RGB, mask and
depth hashes. This can recover geometry missing from an occluded first view; it
does not certify the generated shape, scale, registration or full-clip tracking.
Review the selected source RGB/mask and rerun metric-size and fresh pose checks
before promoting a candidate. The human timeline and camera convention remain
unchanged.

For an initial pose after frame-zero occlusion, `initial_registration` accepts
`--frame-index N`. It asks the same FoundationPose model for proposals at the
native source frame and applies unchanged image, depth and support gates. The
receipt and any accepted seed retain that source frame index. A later accepted
registration alone does not fill earlier frames, certify the occluded interval
or approve full tracking; those require a separately validated temporal merge.

Use `examples/reconstruction.json` and provide a real prompt file. Prompt
object ID 0 is the person and ID 1 is the manipulated object. Example prompt
coordinates are illustrative and must be replaced for your video. Current
scope is one visible person, one rigid manipulated object and a static camera.
There is no automatic support-surface reconstruction in this first adapter.

## Backend environment

The build recipe is `docker/Dockerfile`, using the repository root as context.
Choose and record explicit upstream commits for separately installed SAM2,
SAM 3D Objects and nvdiffrast. The recipe is supplied for reconstruction
environment preparation; **it has not been rebuilt and GPU-validated as part
of this extraction**. Existing compatible environments can also install
`pip install -e . -e modules/reconstruction`.

```bash
docker build -f modules/reconstruction/docker/Dockerfile \
  --build-arg SAM2_REF=YOUR_REVIEWED_COMMIT \
  --build-arg SAM3D_REF=YOUR_REVIEWED_COMMIT \
  --build-arg NVDIFFRAST_REF=YOUR_REVIEWED_COMMIT \
  -t real2sim2real-reconstruction:local .
```

Provide models in the explicit layout in [INSTALL](../../docs/INSTALL.md).
All model acquisition happens outside a run. The stage requests offline
Hugging Face operation. If you split SAM2/SAM3D and tracking into independent
environments, `backend_pythons` maps `segment`, `depth`, `human`,
`depth_alignment`, `metric_mesh`, or `object_motion` to their Python
executables. Install core and reconstruction distributions in each one.

Run with a guarded GPU session only after CPU preparation. The new stage
adapter's contract is CPU tested; end-to-end model inference in this packaging
still needs a short acceptance clip.

## Isolated phase execution and diagnostics

The dispatcher uses only the standard library. Each phase runs in a worker, so
packing and publishing do not accidentally require the calling environment to
contain every model's dependencies. The phases, in order, are:

`video`, `segment`, `pack_masks`, `depth`, `export_human`, `human`,
`depth_alignment`, `metric_inputs`, `metric_mesh`, `object_export`,
`object_motion`, `publish`.

For container integrations, `backend_commands` maps a phase to an **argument
array** such as `["/path/to/guarded-container-wrapper", "--", "python"]`.
The dispatcher appends `-m real2sim2real.reconstruction.backend --phase ...
--request ...`. A wrapper must expose the explicit request/input/output/model
paths at the same absolute paths, install the module in its selected environment,
and enforce container-aware GPU/time limits. It must not mount unrelated user
files or reference data. CPU phases should run without GPU access.

Without a command prefix, `backend_pythons` selects a phase interpreter; otherwise
the calling Python is used. All execution uses argv, never shell interpolation.
`backend_timeout_seconds` bounds each phase (default 7,200 seconds); retain a
separate whole-stage deadline in the enclosing runner.

`commands.json`, `phase_status.json` and `<phase>.log` identify what ran and where
a failure occurred. A failed output is retained for diagnosis and cannot silently
be reused. The final artifact is published only after all phases finish. The
[learning register](../../docs/LEARNINGS.md) distinguishes CPU checks from model
acceptance evidence.

For a full CLI/dependency check in a selected backend environment, use:

```bash
python -m real2sim2real.reconstruction.backend \
  --request /absolute/path/request.json --phase human --check-backend
```

Use a separate preflight output directory. This imports the actual model CLI
with CUDA hidden and requests `--help`; it performs no inference. Check each
model phase this way. Import checks do not prove that all model weights are
available: verify the offline caches separately before starting a long run.
Object generation uses `sam3d_weights/{hf_home,torch_home}`; human/depth workers
use `weights/hf_home` and `weights/sam3d_body/torch_home`. These caches must not be
interchanged, even when both contain a repository with the same model-family name.


### Inspecting the full human surface

The `human` artifact role contains parametric MHR motion and body shape, not
only keypoints. Decode a complete surface without fitting another human model:

```python
from real2sim2real.human_surface import decode_surface
receipt = decode_surface(
    "output/human.npz", "licensed-assets/mhr_model.pt", "review/human_surface"
)
```

This optional CPU operation requires Torch, NumPy and SciPy. It preserves all
frames and uses bounded batches and an on-disk vertex array. `vertices.npy` is
in OpenCV camera/world coordinates in metres; `faces.npy` retains model topology.
An optional proper rigid `world_transform` is applied after native-centimetre
conversion. The receipt pins input/model/output hashes. Partial outputs are not
silently reused, and a completed cache is reused only after hash verification.
This is surface decoding for inspection, not new pose estimation, repair,
retargeting or evidence of physical contact. A translucent display is optional;
it does not change the underlying reconstruction.


### Execution validity and quality warnings

New artifacts also include a `quality` role (`quality_report.json`). It separates
producer-reported image-mask agreement and object motion discontinuities from
`tracking_reliable_frames`. A tracker can finish every frame while image fit is
poor. Missing image evidence is reported as unassessed; it is not a pass.

The configurable `quality_limits` use per-frame bounds (default 8 cm translation,
30 degrees rotation, IoU below 0.40 on more than 10% of frames). Interpret motion
bounds with the reported frame interval and change them explicitly for different
sampling rates/tasks. They are review heuristics, not physical laws. The report
records limits, warning frame IDs and source geometry/tracking hashes; it never
changes motion or approves retargeting/actor training. Human alignment, absolute
metric scale, support and physical contact remain unassessed by this screen.

Compare a fresh output against any retained strong candidate before replacing it.
After a repair changes geometry or poses, recompute image agreement: copying old
tracker scores would describe the wrong trajectory. Downstream matched-mask
comparisons and physical evaluation remain separate from this producer report.

Standalone FoundationPose tracking must use the same explicit settings as the
pipeline: `tracking_runtime.tracking_environment(weights_directory)` validates
both model configurations/checkpoints and returns the weights/cache-independent
runtime settings. Do not rely on the vendor CLI's source-relative weights
fallback. CLI-help checks verify imports; loading the actual checkpoints on CPU
is a separate preflight and neither certifies GPU execution.

### Inspect segmentation prompts before inference

The CPU video phase saves `prompt_review/first_frame_prompts.png` beside
`prompt_review/prompt_review.json`. The receipt pins the actual RGB, source video
and prompt hashes and validates native-pixel coordinates and timeline bounds.
Inspect this preview during CPU preflight before running segmentation. An old
prompt can be inside image bounds yet identify a neighboring person or background;
`semantics_verified=false` makes that limitation explicit. Later-frame and mask
prompts need their own image inspection. Correct misplaced prompts and regenerate
the dependent masks/human/depth/mesh stages while retaining the rejected attempt;
do not weaken downstream geometry checks to accept the wrong segmentation.

After mask packing, `mask_review/contact_sheet.jpg` and `mask_review.json` expose
the propagated **human** and **object** masks separately on source RGB. Both mask
timelines must cover every source frame. Five sampled frames are shown by
default; `mask_review_frames` accepts explicit source-frame indices. The report
pins RGB/prompt/mask hashes and records foreground areas, bounding boxes and empty
samples. Inspect subject identity before geometry inference or after correcting
prompts: a well-populated mask may still follow background. These CPU diagnostics
keep `semantics_verified=false`; they do not lower alignment gates or approve a
reference for training. Corrections invalidate masks, human inference, human-depth
alignment and dependent object geometry/tracking. RGB-only raw depth can be reused
only with verified source/model identity and an explicit derived-file receipt.

For prompts on several source frames, use `review_prompt_frames` with an explicit
frame-index-to-RGB-path mapping. Every annotated frame needs its own native RGB
preview; boxes from a later frame must not be drawn on frame zero. The combined
receipt pins each preview and retains `semantics_verified=false`. Inspect the
actual propagated masks too: a nonempty background mask can satisfy area checks
while corrupting the predicted human and every dependent metric stage.

Mask review also audits foreground counts over **every** source frame. Human
masks below SAM 3D Body's existing 20-pixel input minimum stop at mask packing,
before depth and human inference. The retained JSON lists the offending source
frames and preserves preview evidence. This moves an existing model-input
failure earlier; it does not relax that minimum or claim semantic correctness.
Empty object masks are recorded separately because object visibility can vary.

### Experimental bounded RGB tracking recovery

`python -m real2sim2real.reconstruction.tracking_recovery --help` exposes an
opt-in entry point inside the provisioned reconstruction backend environment
(engine import paths and explicit tracking weights/runtime must be configured,
as for the ordinary object-motion worker). It requires an isolated export,
predicted depth, an independently checked observed frame-zero seed, an explicit
camera-frame support JSON, and a fresh output. The normal backend remains unchanged.

Normal proposals use the existing RGB refiner. Every update is checked against
image overlap/precision/recall, corresponding surface depth, support penetration,
and translation/rotation step limits. On failure, bounded local hypotheses and
fresh depth/RGB registrations compete under the same recovery gates. Budgets are
12 global registrations and 180 recovery frames; failure stops without publishing
an accepted trajectory. Accepted states and failed attempts are retained in JSONL
traces. A caller still owns GPU thermal/deadline supervision and full-clip/native
validation after execution.

With `--continuous-cpu-recovery`, a failed tracker update can fit a bounded pose
from multiple seeds: its highest ranked local proposal, the last accepted pose,
then other motion-bounded proposals. Each fit is checked again by the GPU renderer
for image, depth, support and motion validity. `--max-continuous-calls` limits
the total recovery calls across the clip (default 4, maximum 32); reaching that budget
stops CPU recovery, and the ordinary quality gates remain unchanged. The trace
records each attempted seed and its GPU recheck. A short successful prefix does
not approve the full episode.

The additional opt-in `--continuous-translation-first` requires
`--continuous-cpu-recovery`. It first holds a proposed rotation while fitting
translation, then uses the ordinary full fit if that proposal fails. Final
motion is checked against the actual preceding accepted pose, not against the
unaccepted seed. Every result still faces independent GPU validation. This
mode permits at most two bounded fits per recovery call, so include both stages
when setting a caller's deadline; it does not change default tracking, depth,
image or motion limits. The output records whether staged recovery was enabled.

For thin edge-on silhouettes, `--continuous-image-max-size` can retain a larger
CPU optimization grid (default384, up to4096). The optional
`--continuous-subpixel-radius` searches25 translations within at most2pixels
around a CPU proposal using the actual GPU renderer; default0 disables it.
Both require continuous CPU recovery when changing the defaults, retain all
motion/image/depth/support gates, and remain experimental online options.
See [thin-object tracking and evidence reuse](../../docs/THIN_OBJECT_TRACKING.md).

`recovery_quality` contains the lightweight numerical gates. Their initial
per-frame recipe uses 0.55 tracking IoU, stricter 0.60 recovery IoU, 0.60
precision/recall, 0.10 m median depth error with 100 correspondences, 0.02 m
support penetration, 0.08 m translation and 30° rotation per step. Interpret step
bounds with the source sampling rate. Passing these proposal checks is not
reference, tabletop-contact, physics or actor-training approval. A floor-only
support record must remain explicitly floor-only; it cannot certify a tabletop.

The recovery entry point explicitly binds its exporter to the selected packaged
engine source and checks the observed-seed API. Offline preflight callers should
invoke `tracking_recovery.load_exporter()` too: a successful import alone can
resolve an older container copy and miss an API mismatch. Foreign cached exporters
are rejected instead of silently reused.

### Subject review before dependent inference

The backend now defaults to `mask_review_policy: require_acceptance`.
After `pack_masks`, inspect `mask_review/contact_sheet.jpg`, including the human
and **whole visible target object** at the sampled source frames. A box entirely
inside the image can still select the wrong subject or only part of the target.
The review records all-frame mask hashes and warnings for object disappearance
and area shrinking below half its first visible area. Occlusion and changes of
view can legitimately cause these warnings; they require inspection rather than
automatic rejection or artificial mask filling.

After inspection, write an explicit receipt:

```bash
python -m real2sim2real.reconstruction.mask_acceptance \
  --review OUTPUT/mask_review/mask_review.json --masks OUTPUT/masks \
  --output OUTPUT/mask_review/acceptance.json --reviewer YOUR_NAME \
  --human-accepted --object-accepted
```

If reviewed warnings are present, list their codes with
`--acknowledge object_mask_missing object_mask_shrinks` as applicable.
This is an operator assertion about the visible subjects, not an automatic
semantic judgment or evidence of physical correctness.

`export_human`, `metric_mesh` and `object_motion` verify that acceptance matches
the source-video hash, prompt hash, review hash and **every propagated mask**.
Changed prompts, changed unsampled masks, a different video, or an absent review
invalidate acceptance. Backend exit78 and dispatcher state `review_required`
identify this pause; retained outputs are available for inspection. After approval,
continue the remaining backend phases using the saved request and recorded commands;
do not rerun the whole dispatcher into the same output directory.

Dependency checks (`--check-backend`) and segmentation do not require prior
acceptance. For explicitly exploratory unattended runs, callers may set
`mask_review_policy: unreviewed`; this records `mask_review/unreviewed_run.json`
and leaves subject identity unaccepted. This opt-out does not approve geometry,
retargeting or task success. A custom receipt path can be set with
`mask_acceptance`, resolved relative to the configuration directory.

For a lost pose after occlusion, `tracking_inputs.create(...,
frame_indices=...)` can export an isolated forward or backward source interval.
The indices must be contiguous integers in source bounds. RGB, object/human
masks and all depth timelines are sliced together; tracker-local names start at
zero. `source_frame_mapping.json` and the input receipt preserve the original
source frame IDs, full source video hash and parent artifact hash. The full
video is retained as provenance, not relabeled as a shortened video. Old poses
are never exported to registration. A successful window is not a full-clip
repair: joining windows requires explicit continuity and occlusion review.

Rebuilt tracking inputs preserve source vertex colours/UV appearance when the
source mesh matches the metric geometry vertices and triangle indices. The
receipt records the source mesh hash and appearance status. For RGB cues such
as a painted panel, request `require_appearance=True`; missing or mismatched
appearance then fails explicitly. Rebuilding a geometry-only mesh must not
silently be treated as preserving the feature needed for RGB registration.
Appearance correspondence does not itself validate pose or tracking.

The FoundationPose export loader also preserves existing vertex/face colours.
Only texture visuals need conversion to vertex colours. Calling a missing
to_color method on ColorVisuals previously entered an exception fallback that
replaced genuine painted features with uniform grey. A failed texture conversion
now raises explicitly; grey is used only when source appearance is absent.
Check the actual loaded tracker mesh, not just the exported GLB, when diagnosing
RGB orientation drift on nearly symmetric objects.

For a separately reviewed source-visible colour cue, `tracking_recovery` accepts
an optional `--appearance-guard` JSON. It declares uint8 RGB channel differences
and bounds, a minimum visible pixel count, source-support precision, pixel
tolerance, a unit camera-frame recovery axis, source evidence and
`ground_truth_used: false`. Native GPU triangle rasterization checks the visible
interpolated mesh colours against that frame's RGB/object mask, excluding human
occlusion. A mismatch triggers bounded recovery under the original geometric
and temporal gates. It never makes silhouette/depth checks optional. Hidden
model features remain explicitly uncertain; their absence does not prove the
orientation correct. Keep feature thresholds and observed-colour mesh provenance
in the downstream run manifest, and check other visible cues independently.

For declared sensitivity experiments, the guarded tracking CLI accepts
`--depth-error-m` (metres). The default remains 0.10 m. A larger configured value
is explicitly marked as threshold relaxation and serialized into the recovery
protocol; other acceptance checks remain unchanged. This is not an automatic
fallback or reference approval. Compare full-clip image/motion evidence and
preserve stricter-run results before adopting a tolerance variant.

`tracking_proposals` is an explicitly unaccepted RGB tracking mode for downstream
multi-frame fitting. It requires a hash-bound accepted initial-registration
report and seed. Its outputs and receipt are marked `proposal_only`; the basic
tracker's confidence flag does not establish independent trajectory validity.
Use the reconstruction-repair temporal-window composition before considering
these predictions for a reference.

For objects with thin grasp features, `options.mask_feature_checks` can require
native-image mask coverage before `metric_mesh` or `object_motion`. Each entry
contains a source `frame_index` and named `features`, each with an integer
`xyxy` rectangle and `min_fraction`/`max_fraction`. For example, require foreground
on a rail and grip and background in the open gap between rails. `pack_masks`
saves fractions and mask hashes to `mask_review/feature_coverage.json`; dependent
stages recheck the actual pixels. The checks are optional and do not substitute
for direct subject review, metric calibration or physical evaluation. Select a
clearer `mesh_frame_index` when the grasp feature is poorly visible initially;
source frame numbering and matched RGB/mask/depth must remain unchanged.


### Zero-copy tracking windows

`tracking_inputs.virtual_slice_h5` builds forward or backward contiguous HDF5 windows without duplicating image/depth payloads. It records native-frame mappings and relative source dependencies. Keep the retained source files available and mounted with the virtual window; moving only the small virtual file is insufficient. This is input routing, not interpolation or new evidence.
# Optional external recovery

Normal tracking can optionally request fresh Do as I Do proposals after a failure.
The upstream checkout/environment/weights remain separate. The portable adapter,
shared-camera inputs, fixed-scale exports and independent acceptance live here;
the default pipeline is unchanged. See [external object recovery](../../docs/EXTERNAL_OBJECT_RECOVERY.md).

For offline use, `recovery_lookahead.qualify` can require a bounded sequence of
successful normal updates before admitting a recovery. It retains exact accepted
states and rejected-branch diagnostics; image overlap alone is insufficient.
`daid_shape` qualifies the fixed mesh and source inputs on CPU, then encodes a
mesh-specific cache under an external GPU supervisor. These are optional helpers,
not an automatic replacement of the existing tracking pipeline.
