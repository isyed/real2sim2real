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
7. `review`: create an offline HTML slider with image projections and a rotatable
   metric world view. With no 3D hands, `contacts-preview/` and `review-preview/`
   preserve a clearly labelled partial result without occupying final outputs.
8. `publish`: verify shared timestamps and coordinate composition, then publish
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
