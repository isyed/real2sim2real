# Egocentric reconstruction

An independently installed stage for calibrated moving-camera captures. The
initial implementation reconstructs one rigid object with metric stereo depth,
background RGB-D odometry, SAM2, SAM 3D Objects and FoundationPose. It publishes
`object-reconstruction`, deliberately distinct from full-body `reconstruction`.
The optional `metric-interaction` consumer adds a support surface and articulated
hands from the observed images and stereo. Neither output represents full-body
motion or a robot controller.

Install the core and this module. GPU workers additionally use the reconstruction
module and separately provisioned model runtimes. Importing or planning does not
load models. Source, images, weight directories and GPU session are explicit
configuration; no neighboring checkout is imported.

```bash
python -m pip install -e . -e modules/egocentric-reconstruction
python -m pip install -e 'modules/reconstruction-evaluation[plots]'
```

Start with [examples/config.json](examples/config.json); every path, image and
prompt is a placeholder. Prompts use the prepared image grid, after undistortion
and resizing. Add prompt entries for occluders if using `occluder_ids`. Choose a
background ROI with static texture; moving hands and objects are excluded by
their masks. This baseline needs calibrated, synchronized horizontal stereo;
uncalibrated monocular input is not accepted by this stage.

Workers mount this checkout's reconstruction engines. The chosen images must
provide their dependencies and compiled extensions. For FoundationPose,
`native_python_paths.track` names the directory containing the `mycpp` package,
not its `build` subdirectory. Model preflight checks the native `cluster_poses`
entry point as well as imports and local weight files. These checks do not prove
GPU execution; a bounded inference experiment is still required.

The `calibrated-capture` artifact contains `capture.json` with schema
`real2sim2real.calibrated-capture/1`, OpenCV cameras, metres, consecutive source
frame IDs, capture timestamps, and named video streams. Each stream declares
video SHA-256, image size, intrinsics, distortion and `rig_from_camera`. The
manifest selects `target_stream` and ordered `stereo_pair`. Calibration must
already describe the released pixel orientation; acquisition-specific rotation
and camera-name mapping belong in the capture adapter.

The camera backend refines relative stereo pose on declared calibration frames
with fixed baseline length. Separate frames validate epipolar errors. It retains
both the original and candidate transforms; rejected calibration stops the run.
Depth is reprojected with a z-buffer into the undistorted target image. Unobserved
depth stays zero. SAM 3D receives the same pointmap and intrinsics as tracking.

`r2s2r-ego --config private-config.json --phase PHASE` supports `calibrate`,
`prepare`, `segment`, `camera`, `mesh`, `track`, `publish`. Use `--check` on the
three model phases for CPU import checks. Inspect masks and geometry between
phases. The Docker worker receives only source, model stores and its prediction
directory; do not place reference ground truth in that directory.

Each model run records source hashes, arguments and telemetry. The local guard
requires an active bounded 1500 MHz clock lease, pauses at 82 C, resumes at 74 C,
and stops at 83 C. The guard owns the Docker container by immutable ID and cleans
it up on failure. The privileged helper restores clocks independently on expiry
or a completion marker. CPU preparation and scoring require no GPU lease.

Camera and object failures remain explicit. Object tracking saves rejected raw
diagnostics but never advances from a rejected pose. Only complete accepted
trajectories can be published. Source-time velocities are evaluated in a fixed
world frame. Initial world origin is the first target camera, not motion-capture
world. Ground-truth alignment is exclusively an evaluator operation.

Each model attempt has an immutable `attempts/PHASE/ID` directory containing its
request, source hashes, log and thermal telemetry; `PHASE.latest.json` points to
the latest attempt. Existing model results are not overwritten. Use a new output
directory for another hypothesis. To extend a validated prefix, set
`reuse_mesh_from` to that prefix directory and run the `mesh` phase after preparing
and segmenting the extension. It copies the mesh only if capture, image grid,
conditioning RGB/depth/mask and conditioning frame match, and records their hashes.

Tracking defaults to five refinement iterations with no retries. Set
`model_options.tracking_iterations` and `model_options.retry_iterations` for a
bounded refinement experiment, for example `5` and `[1, 2]`. Every retry starts
from the last accepted pose and passes the same silhouette, depth and world-motion
gates. A failed candidate never seeds a retry or the next frame. All candidates
are retained in `tracking_quality.json`; `raw_camera_from_object` preserves the
first proposal and `camera_from_object` contains only accepted results. Complete
coverage does not establish geometric accuracy: reference scoring and image
review remain necessary, especially for thin or weakly textured objects.

`real2sim2real.egocentric.reuse.fork_tracking_inputs(source, destination)` copies
only observations, camera motion and the fixed mesh into an empty hypothesis
directory, with hashes. It excludes predicted object poses and copies real files
instead of sharing writable files with the parent. Use it to compare tracking
settings without regenerating meshes or changing the original experiment.

Run the clock helper from a terminal with permission to manage the GPU:

```bash
sudo python src/real2sim2real/runtime/temporary_gpu_clock.py \
  --session-dir /data/gpu-session --mhz 1500 --seconds 28800
```

At the end of all stages, create `/data/gpu-session/clock_session.done` and check
that `clock_session.json` reports restoration. The helper also restores clocks
on expiry or interruption; the inference guard stops if the lease disappears.

## Metric interaction reconstruction

Install the optional CPU dependencies with
`pip install -e 'modules/egocentric-reconstruction[hands2d,contact]'`.
Start from [examples/interaction.json](examples/interaction.json), replacing the
placeholder paths and support selection. The `reconstruction` directory is an
existing accepted object artifact; a separate `output` preserves that baseline.
All coordinates are metres and column-vector transforms. Pixel coordinates use
the same undistorted image grid as the object pipeline.

Run `python -m real2sim2real.egocentric.interaction --config CONFIG --phase PHASE`
in this order, inspecting the prefix before extending the frame count:

1. `support`: select static tabletop pixels using a declared ROI/polygon and
   optional HSV range, exclude foreground masks, and robustly fit stereo points
   from several views. Validate on disjoint frames. Export one rigid
   `upright_from_world` transform, the transformed camera/object motion, and a
   closed finite collision slab with its top at z=0. Up assumes the chosen table
   is level; it is not an IMU gravity measurement. The convex boundary covers
   observed support; unseen table extent is unverified. Report raw outliers as
   well as consensus. Pooling views does not correct camera drift.
2. `hands2d`: run MediaPipe on the whole image and hand crops on CPU. Associate
   detections with explicit left/right mask IDs. Save 21 image landmarks and
   rejected/missing observations. MediaPipe's learned 3D prior is retained
   separately and is not treated as a measured world-space hand.
3. `hands-check`: load the pinned HaMeR checkpoint and MANO asset on CPU.
4. `hands`: run HaMeR through the existing clock lease and container guard.
   Preserve articulated MANO poses and use a constant median shape per hand.
   Raw vertices are wrist-relative in camera axes; predicted weak-perspective
   camera values are not used as calibrated metric translations. The worker
   intentionally leaves missing 2D observations missing.
5. `align-hands`: fit each articulated prior to image landmarks and measured
   skin surfaces, with bounded rotation/translation and one constant scale per
   hand. Surface residuals use exact triangle distances. Spatially interleaved
   stereo pixels validate the fit; these still share sensor/model errors and
   are not independent ground truth. Transform accepted hands using the same
   `upright_from_world @ world_from_camera` as the object. Preserve rejected
   proposals separately; accepted arrays use NaN for missing frames. There is
   no contact constraint, gap filling or hidden trajectory rescaling.
6. `contacts`: report finite-table clearance, hand–object proximity candidates,
   fingertip gaps and penetration. Signed penetration is available only for a
   closed, consistently wound object mesh. Geometry is not moved to manufacture
   contact. Contact diagnostics do not prove physical contact or control success.
7. `score`: save `quality_summary.json` with the full frame denominator, missing
   detection/model counts, rejected-fit reasons, per-hand metric coverage and
   contact statistics. Report fit errors both for accepted estimates and for all
   scored candidates, so discarding failures does not hide their errors. These
   are image/depth consistency scores, not ground-truth joint accuracy.
8. `review`: create an offline HTML slider with image projections and a rotatable
   metric world view. With no 3D hands, `contacts-preview/` and `review-preview/`
   preserve a clearly labelled partial result without occupying final outputs.
9. `publish`: verify shared timestamps and coordinate composition, then publish
   `metric-interaction-reconstruction`. Both hands need accepted estimates;
   per-frame gaps remain explicit and `complete=false`. The artifact is marked
   `control_ready=false`. It cannot be passed off as full-body reconstruction.

CPU invocations record source/configuration hashes; the input receipt includes
RGB, depth, masks, geometry and mesh hashes. GPU workers mount only this checkout,
declared model stores, the output, and explicitly declared read-only observations.
Never mount reference scans or reference motion for inference. New hypotheses
require new output directories; completed phase outputs are not overwritten.

The hand image is built from
[docker/hamer.Dockerfile](docker/hamer.Dockerfile), with HaMeR pinned at commit
`3a01849f4148352e9260b69bf28b65d1671a4905`. Its default base is the public PyTorch
CUDA 12.8 image. A compatible cached runtime can be supplied via `BASE_IMAGE` and
`INSTALL_DEPENDENCIES=0`; the import check must still pass.

```bash
docker build -f modules/egocentric-reconstruction/docker/hamer.Dockerfile \
  -t r2s2r-hamer:3a01849 modules/egocentric-reconstruction/docker
```

Use the [official HaMeR asset instructions](https://github.com/geopavlakos/hamer)
to populate the configured weight directory with `_DATA/hamer_ckpts/` and
`_DATA/data/mano_mean_params.npz`. The official archive does **not** contain
`MANO_RIGHT.pkl`: obtain MANO v1.2 through the
[MANO project](https://mano.is.tue.mpg.de/) and place that file in
`_DATA/data/mano/`. Model files remain outside Git. The CPU landmark model is the
[MediaPipe hand-landmarker asset](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task).

For a controlled prefix/full comparison, optionally set
`model_options.fixed_shape_from` to the prefix's `hands_raw/shape.json` and
`metric_hands.fixed_scale_from` to its `hands_metric/quality.json`. Include the
prefix in `read_only_inputs`. The longer sequence can fit one static support
plane from more views, with disjoint validation frames; this may change the
bundle's global upright transform but never the source camera-relative motion.

### Reading the interaction review

The coloured skeletons on the image are accepted **3D joints projected into the
camera image**. Their finger bends show articulation. The world panel displays
those same joints and hand surfaces beside the object and table in metres.
Grey skeletons are **2D landmarks only**: image positions exist, but an accepted
metric hand is missing. A missing frame is kept invalid rather than filled with
the previous pose. Passing the checks means agreement with configured image,
depth and motion limits; it does not establish ground-truth hand accuracy.

A gap in the hand trajectory means an unavailable estimate at that time. A
hand–object gap is a spatial distance between surfaces. Contact diagnostics use
the latter: proximity suggests a contact candidate, while penetration indicates
overlapping reconstructed geometry. They do not simulate forces or certify a
grasp. A prefix is the short initial interval reviewed before the full sequence.

Dense object meshes use bounded centroid/radius searches for exact point-to-face
distances, with bounded projection batches. Tests compare this query with an
exhaustive triangle search, including far-away points and unequal triangle sizes.
CPU fitting and contact runs can use `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`
to avoid oversubscribing small geometry operations.
For dense meshes, the optional `contact-fast` extra adds the Embree CPU ray
accelerator. Select `contacts.ray_backend` as `auto` (default), `triangle` or
`embree`; the actual backend is recorded in the diagnostic report. An explicit
`embree` request fails if its dependency is unavailable. The geometric distance
query and contact thresholds are unchanged. See the
[Trimesh ray backend documentation](https://trimesh.org/trimesh.ray.ray_pyembree.html).

### Refining rejected hand intervals on CPU

Use [the refinement example](examples/hand-refinement.json) with a verified
`metric-interaction-reconstruction` parent and a new output directory. The parent
and candidate must have exactly the same frame IDs and timestamps. Inspect the
short prefix before extending the same optimizer settings to a longer parent.

Run these phases sequentially with `r2s2r-interaction --config CONFIG --phase PHASE`:
`audit-hands`, `refine-hands`, `contacts`, `score`, `compare-hands`, `review`,
`publish`. The audit locates rejected/missing intervals and per-joint image
disagreement. Refinement uses the existing local HaMeR container to load only the
licensed MANO model, with networking and GPU access disabled. It has a finite
CPU deadline and removes its owned container on completion or timeout. It does
not reload the image encoder or require a GPU clock lease.

The optimizer fits local joint rotations and a small rigid correction using 2D
landmarks and the fit half of stereo surface points. Defaults bound each local
joint correction to 30 degrees, the camera-space rotation correction to 15
degrees, and translation to 3 cm. Shape coefficients and metric scale stay
fixed. The model must reproduce the retained raw vertices/joint convention before
optimization. Missing detections stay missing, and accepted parent geometry is
copied exactly. Optional `hand_refinement.frame_ids` limits the attempted frames.

Admission preserves the parent's image, held-out depth and wrist-speed limits.
It additionally requires at least 10% improvement in median or p90 image error,
no image metric worsening by more than 0.5 px, no held-out median depth regression
over 1 mm, and no held-out p90 regression over 3 mm. New proposals with object or
table penetration over 3 mm are rejected; signed object penetration is available
only for a closed, consistently wound mesh. Contacts are checked after fitting;
they do not pull fingers onto the object. Temporal admission examines newly
introduced transitions and never replaces an accepted parent anchor.

`hand_comparison.json` reports paired residuals for recovered frames, remaining
gaps, new temporal edges and contact consistency. These image errors are **fit
residuals against the input detector**, not an independent hand accuracy score.
Held-out depth is spatially disjoint but still comes from the same stereo capture.
Occluded fingers and erroneous 2D landmarks remain limitations, and the artifact
stays `control_ready=false`.

Refinement preserves source hashes, failed proposals and the baseline quality
report. `hands.npz` adds `articulation_delta_rad` and `refinement_accepted`.
For admitted refinements, `fits` contains the updated rigid transform; interpreting
it with the raw model also requires the saved local articulation delta. Downstream
consumers should use the explicit accepted camera/world vertices and joints.

### Visibility-aware temporal fitting and missing-observation probes

[The temporal example](examples/temporal-hands.json) starts from a published,
independently refined interaction. Run `probe-hands2d` (optional),
`temporal-hands`, `contacts`, `score`, `compare-temporal-hands`,
`assess-readiness`, `review`, `publish` sequentially. Use a new output directory
and inspect a prefix before extending the same settings. Everything in this
sequence runs on CPU; the model container has no network or GPU access.

`temporal-hands` may adjust only the parent's `refinement_accepted` observations.
Original unrefined anchors, missing estimates, detections, shape, scale and scene
geometry remain fixed. It does not repeatedly smooth already temporally refined
outputs. Neighbor targets use the actual capture timestamps and the fixed world
frame. Both immediate neighbors must be accepted, with each time step at most
`max_step_s` (default 0.1 s). No missing hand is filled. Future observations make
this an offline reconstruction stage, not an online controller.

Image weights use the hand/object masks and only the fitting checkerboard of
stereo depth. Unsupported landmarks retain a small weight. This is a visibility
proxy, not calibrated confidence or an assertion that an occluded finger is
correct. MANO corrections are bounded to 8 degrees per joint, 3 degrees of rigid
rotation and 5 mm translation by default. Backtracking tries fractions 1, 1/2,
1/4, 1/8 and 1/16 on the rotation groups; it never blends mesh vertices.

The original absolute image/depth gates still apply. Median and p90 image errors
may each worsen by no more than 0.5 px; held-out depth may worsen by no more than
1 mm median or 3 mm p90. New object/table penetration above 3 mm fails. Changes
are accepted together when they affect common temporal triplets: acceleration
RMS must improve by at least 5%, its peak cannot grow by more than 5%, and speed
p90/maximum cannot grow by more than 0.05/0.1 m/s. These are comparative regression
checks, not robot-specific dynamic feasibility limits. Failed proposals fall back
to the parent without reducing coverage. Every trial and backtracking fraction
is retained. The full run can make different decisions near a prefix boundary
because additional observed neighbors are available.

`probe-hands2d` tests missing detections in tight, rotated crops. Nearby detections
provide search boxes only; reported landmarks are fresh image-model outputs.
Candidates need the original 65% hand-mask overlap, at most 30% overlap with the
other hand, and agreement from at least two distinct crop/rotation requests
(median <=10 px and p90 <=20 px). A probe result is not automatically admitted to
metric reconstruction. Repeated model agreement is not independent accuracy
validation, and unsupported occluded joints remain an open issue.

`temporal_comparison.json` compares identical accepted frame sets, reports
acceleration and speed in metres/seconds, and verifies immutable anchors.
`readiness_assessment.json` lists remaining metric gaps, contiguous observed
intervals, contact failures and requirements still needing validation. It also
reports a clearly labelled optimistic coverage bound for the new 2D candidates;
those candidates never count as recovered 3D hands. Complete coverage alone does
not certify control readiness: a particular robot still needs retargeting,
collision and dynamic replay validation.

## Compare detector-independent video hand observations

`r2s2r-hand-experiment` compares a detector-gated HaMeR baseline with HaMeR and
HaWoR using identical detector/mask-derived crops. See
[`examples/video-hands.json`](examples/video-hands.json). The frame window is a
zero-based start position plus a count; original capture frame IDs, timestamps
and world transforms are retained. Selecting a later window does not create a
new camera-world origin.

The hand detector uses the WiLoR checkpoint selected by the pinned HaWoR fork.
Physical left/right identities are assigned through the supplied masks, with
one detector box assigned to at most one hand. A mask box is a fallback when
localisation fails. Neither box availability nor a network prediction counts
as a validated metric observation. Missing crops remain missing; the adapter
does not invoke HaWoR's motion infiller.

The HaWoR backend uses its temporal camera-space network in blocks of 16 images.
Boundary images are repeated only as network padding and are never exported as
extra observations. Both estimators use the same HaMeR crop preprocessing,
normalisation, crop expansion, fixed MANO shape and calibrated metric scale;
input tensor hashes verify that their crops match. This is a controlled adapter
comparison, not an unmodified reproduction of HaWoR's complete demo pipeline.
HaWoR receives the supplied principal point (reflected correctly for the left
hand) and a scalar network focal equal to the geometric mean of fx/fy. Metric
fitting and evaluation always use the full calibrated intrinsic matrix.

Provision the optional runtime separately. Clone
`https://github.com/malik-group/HaWoR.git` outside this repository and check out
`2c3fa0c06e54a1b192c1ff692eaa89d526cd6317`, the revision used by do-as-i-do.
Build `docker/hawor.Dockerfile` with that checkout as context, using an existing
HaMeR runtime as `BASE_IMAGE`. The image checks the source revision and preserves
its upstream notices. Do not put videos, weights or licensed MANO models in the
build context. The model store contains `hawor.ckpt`, `model_config.yaml` from
`ThunderVVV/HaWoR`, and `detector.pt` from `rolpotamias/WiLoR`. Record immutable
model revisions and file hashes. The existing HaMeR store supplies MANO_RIGHT.
The adapter uses the same mirrored right-hand MANO convention as the baseline.
Optional backend/model licenses remain separate from this project's license.

Run preparation and import checks before the bounded GPU stages:

```bash
r2s2r-hand-experiment --config /work/video-hands.json --phase prepare
r2s2r-hand-experiment --config /work/video-hands.json --phase detector --check
r2s2r-hand-experiment --config /work/video-hands.json --phase hawor --check
# Requires an active clock lease; each container has a thermal guard and deadline.
r2s2r-hand-experiment --config /work/video-hands.json --phase detector
r2s2r-hand-experiment --config /work/video-hands.json --phase hamer
r2s2r-hand-experiment --config /work/video-hands.json --phase hawor
# CPU fitting and review; contact-fast is optional.
timeout 2400 r2s2r-hand-experiment --config /work/video-hands.json --phase fit --variant baseline
timeout 2400 r2s2r-hand-experiment --config /work/video-hands.json --phase fit --variant hamer
timeout 2400 r2s2r-hand-experiment --config /work/video-hands.json --phase fit --variant hawor
timeout 300 r2s2r-hand-experiment --config /work/video-hands.json --phase review-inputs
# Supply evaluation-only visible landmark annotations, then:
timeout 300 r2s2r-hand-experiment --config /work/video-hands.json --phase evaluate
```

`inputs/archived_baseline.npz` retains the previous accepted reconstruction.
`baseline/` re-evaluates its original raw predictions with the common fitter so
changes to fitting are distinguishable from changes to detection/model input.
The new fitter uses measured stereo pixels, visible automatic landmarks and
silhouette residuals. A disjoint depth checkerboard measures held-out surface
consistency. It fixes shape/scale/camera/object/table; contact is evaluated after
fitting. A hypothesis can pass geometry checks without every joint being visible.
Those results are labelled `geometry_supported`, never silently substituted for
the archived baseline. Rejected hypotheses are retained for inspection.

Evaluation produces `review/review.html`, `comparison.json` and a standalone
`annotations.html` editor. Where a calibrated capture is provided, it exports
undistorted secondary images and projects hypotheses using the existing stereo
extrinsics. Visual labels bind source frame, physical hand, joint, camera and
image hash. They live outside the worker mounts and are never used for fitting.
Annotate only clearly visible joints, record uncertainty and reviewer identity,
and do not label hidden joints from model overlays. Assistant-inspected anchors
must be marked as such, with `human_verified: false` until a person reviews them.

Reports separate inferred poses, geometry support, losses/recoveries relative
to the archived baseline, automatic landmark residuals, held-out depth, identity,
contact, temporal measurements and visual-anchor errors. These are consistency
diagnostics and sparse image evidence, not complete 3D ground truth. The command
never promotes a candidate or declares control readiness. Use a new output or
`review_directory` when repeating an evaluation; preserve prior hypotheses.
## Point-guided articulation experiments

`r2s2r-point-hand-experiment` adds optional BootsTAPIR observations to a completed
hand-observation comparison. It retains the calibrated camera/object/table
world and fixed MANO shape and scale. Configure a new output directory using
[`examples/point-hand-experiment.json`](examples/point-hand-experiment.json).
Frame IDs in the seed and held-out lists are **source IDs**, while `frame_start`
is the reconstruction array index. The example assumes IDs start at zero.

```bash
r2s2r-point-hand-experiment --config /private/experiment.json --phase prepare
r2s2r-point-hand-experiment --config /private/experiment.json --phase track --check
r2s2r-point-hand-experiment --config /private/experiment.json --phase track
r2s2r-point-hand-experiment --config /private/experiment.json --phase refine --check
r2s2r-point-hand-experiment --config /private/experiment.json --phase refine
r2s2r-point-hand-experiment --config /private/experiment.json --phase evaluate
```

Prepare the run output directory as the normal user before starting its GPU
lease helper. `track` requires the configured GPU lease and thermal guard;
`refine` uses a CPU-only container with a finite deadline. The other phases run
under bounded CPU subprocesses. Check phases load dependencies/checkpoints on
CPU without running image inference. Existing outputs are never overwritten.

Build [`docker/tapir.Dockerfile`](docker/tapir.Dockerfile) with a separately
downloaded `malik-group/tapnet` checkout at
`f2f8888a803051ea42cd79c89cfca6c25dfeae77` as the Docker context. The original
Apache notices remain in that checkout/image; no upstream code is vendored or
imported from a neighbouring project. Put the official
[`bootstapir_checkpoint_v2.pt`](https://storage.googleapis.com/dm-tapnet/bootstap/bootstapir_checkpoint_v2.pt)
under `weights.tapir`; record its digest and download provenance privately.
Build [`docker/hand-refinement.Dockerfile`](docker/hand-refinement.Dockerfile)
for CPU MANO/Embree refinement, using the existing HaMeR image as the base.
The licensed MANO model remains separately supplied under `weights.hamer`.

The pipeline selects visible automatic 2D detector seeds from explicitly
selected frames. Evaluation frames are excluded. Each hand uses one fixed
image crop over the complete window. Tracks must pass visibility/uncertainty,
physical hand masks, a return track re-queried at another time, and agreement
between at least two distinct seed frames. Occluded/drifting observations stay
missing. Raw tracks, return queries and rejection evidence remain available.

The articulation fitter accepts partial landmarks with zero weights for missing
joints. It makes a fresh rigid initialization from fitting depth and tracks,
then fits bounded MANO rotations and a small wrist correction. It never loads
held-out detector labels or visual annotations. `fixed_betas` reconstructs the
HaWoR geometry; its per-frame `betas` field contains unused network shape
predictions. MANO mesh/joint reconstruction is checked before optimization.

Only rejected baseline intervals are eligible for candidate additions. Separate
diagnostic fits at held-out frames allow checks where the baseline is already
accepted; those fits never replace baseline anchors. The diagnostic artifact
retains rejected geometry with explicit support masks. Evaluation verifies that
all accepted baseline geometry and source files are unchanged, reports held-out
2D detector errors, sparse visual annotations in calibrated views, reserved
depth and contact diagnostics, and writes an HTML comparison.

Tracker agreement is fitting evidence, not independent ground truth. Surface
appearance tracks may not follow anatomical joint centres, and multiple seeds
can share detector bias. Held-out detector labels remain correlated with the
seed detector; stereo samples are spatially correlated too. Manual review and
robot-specific replay remain separate acceptance steps. The experiment always
reports `promoted: false` and `control_ready: false`.

### Sequence continuity candidates

After pointwise fitting, copy the experiment configuration, set a fresh
`candidate_directory` and `review_directory`, and add `sequence-point-hands`
to `images` and `timeouts` (the CPU refinement image can be reused). Then run
`--phase sequence`, followed by `--phase evaluate`. `--check` checks the CPU
MANO/contact dependencies. Every sequence candidate starts from the retained
pointwise `refined/hands_metric` result; candidates cannot be chained silently.

`sequence_optimizer.context_frames: 0` changes recovered frames only, using
fixed measured neighbors. A positive value also proposes changes to that many
surrounding frames in a separate candidate. The source baseline is immutable.
Missing observations and capture-time gaps split components. Shared temporal
triples join neighboring components so boundary constraints remain consistent.

The CPU optimizer batches MANO poses and minimizes image/depth residuals with
velocity/acceleration penalties at actual capture times. Joint, rigid rotation
and translation corrections are bounded relative to the retained poses; shape,
metric scale, cameras, object motion and tabletop remain fixed. Default bounds
are 30 degrees, 10 degrees and 15 mm respectively. `anchor_data_weight` can
increase fitting evidence at surrounding baseline frames. A positive
`boundary_velocity_weight` constrains previously observed baseline/baseline
edges, never the recovered/baseline jumps being repaired.

An interval is admitted atomically only after geometry, contact and motion
checks. Validation depth and evaluation labels never enter the objective.
Detector-based checks can still be correlated with the baseline used for
initialization/anchors; they must not be presented as independent ground truth.
Rejected interval hypotheses remain in `diagnostic_*` / `proposed_world_*` arrays;
the primary arrays retain their parent. Reviews distinguish those hypotheses
from the admitted result. Complete MANO poses are exported and checked against
retained meshes; old pointwise deltas are renamed `parent_local_delta_rad`.

Sequence admission establishes relative consistency improvement, not readiness
for control. Inspect absolute finger motion at joins, conflicting observations
and held-out image checks before retargeting or physical replay.

### Cross-camera observation audit

`python -m real2sim2real.egocentric.hand_anchor_audit --config audit.json` compares
existing hand hypotheses with synchronized undistorted review images. It runs
on CPU and writes a fresh `audit.json` and `review.html`; it never updates poses.
Generate review inputs with `hand_review_inputs.prepare_review_inputs`, using
the original capture manifest alongside its videos. Source video hashes,
capture timestamps and the accepted reconstruction calibration are checked
during preparation. Video decoding requires the declared `av` dependency.

The audit config declares `views` (directory containing `views.json`),
`observations` (point-track NPZ), optional evaluation-only `annotations`,
`frame_ids`, two `secondary_views`, a fresh `output`, and `variants` mapping
display names to `{ "path": "/data/hypotheses.npz", "joints_key": "camera_joints" }`.
Use `diagnostic_camera_joints` explicitly to inspect a rejected proposal.
Candidate arrays must share the observation frame IDs and capture timestamps.
Example configurations and image assets for individual experiments belong in
downstream private repositories.

The generic `multiview_points` helpers search along calibrated epipolar lines
over a declared depth interval, using normalized patch correlation, uniqueness
and reciprocal matches. Defaults are 0.2–1.5 metres, patch radius 6 pixels,
correlation ≥0.85, uniqueness margin ≥0.08 and cycle error ≤2 pixels. Two
secondary matches are required for a three-view surface hypothesis; inconsistent
pair depths (>30 mm), reprojection (>2 px), small ray angles (<1 degree) or
high one-pixel depth sensitivity (>20 mm) reject it. These are explicit
diagnostic thresholds, not calibrated confidence estimates. `matching` can
override the search/patch parameters.

The audit also flags distinct observed fingertips within 8 pixels, which can
indicate occlusion or ambiguous identity. It does not relabel those fingers or
turn missing fingers into measurements. Appearance matches establish neither
anatomical joint centres nor control readiness. Secondary views overlap with
the stereo-depth source; this is a consistency audit, not independent 3D ground
truth. Approximate visual labels stay separate from automatic matching. Source
image, input-array and implementation hashes accompany every report.
