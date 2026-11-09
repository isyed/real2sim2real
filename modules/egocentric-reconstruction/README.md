# Egocentric reconstruction

An independently installed stage for calibrated moving-camera captures. The
initial implementation reconstructs one rigid object with metric stereo depth,
background RGB-D odometry, SAM2, SAM 3D Objects and FoundationPose. It publishes
`object-reconstruction`, deliberately distinct from full-body `reconstruction`.
Hand motion and robot control are separate consumers and are not fabricated by
this stage. This first implementation is an object-reconstruction baseline.

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

Run the clock helper from a terminal with permission to manage the GPU:

```bash
sudo python src/real2sim2real/runtime/temporary_gpu_clock.py \
  --session-dir /data/gpu-session --mhz 1500 --seconds 28800
```

At the end of all stages, create `/data/gpu-session/clock_session.done` and check
that `clock_session.json` reports restoration. The helper also restores clocks
on expiry or interruption; the inference guard stops if the lease disappears.
