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
