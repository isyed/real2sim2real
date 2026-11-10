# Testing the reusable modules

Run these commands from the repository root with Python 3.11 or 3.12:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -e '.[test]' ruff
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m pytest -q
.venv/bin/ruff check --select F src/real2sim2real \
  modules/*/src/real2sim2real tests modules/reconstruction-repair/tests
git diff --check
```

Pytest adds the module source directories to its import path. Tests use synthetic
fixtures and temporary output directories; videos, model weights, robot assets,
private repositories and GPUs are not required for this suite. The `test` extra
includes the CPU geometry, image, HDF5, Parquet and checkpoint-format libraries.
Optional simulation/USD, Torch, RoboSimStudio and Embree checks report explicit
skips when their libraries are not installed. A skip is not a successful model
or simulation validation.

Simulation and actor tests also run on CPU in an independently provisioned
FlashCHORD runtime. Use the same repository sources and disable GPU execution:

```bash
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /path/to/runtime/bin/python -m pytest -q \
  tests/test_actor_warm_start.py tests/test_approach_heading.py \
  tests/test_reference_body_pull.py tests/test_reference_denoising.py \
  tests/test_sim_multiccd.py tests/test_action_imports.py \
  tests/test_bounded_approach.py tests/test_residual_controller.py \
  tests/test_world_static_support.py tests/test_frozen_render_meshes.py \
  tests/test_policy_clearance.py
```

The runtime must provide its compatible Newton, Warp, JAX/Flax/Optax and
checkpoint libraries. Torch motion-schema interoperability is checked separately
with `tests/test_motion_interop.py` in a retargeting environment. Model inference,
training and GPU simulation require their own authorized, bounded acceptance
runs; passing the CPU suite does not validate physical grasp success.

Keep test module filenames unique across both test directories. Pytest's default
import mode uses those names as module keys. Numerical utilities should remain
importable without optional model or simulation libraries; subprocess regression
tests enforce this boundary.

The GitHub Actions CPU workflow runs the lightweight suite and undefined-name /
unused-import checks on both supported Python versions. It does not download
models or start GPU inference.

Hand refinement admission, immutable baseline comparisons, missing-frame handling
and CPU worker cleanup are covered by `tests/test_hand_refinement.py` and
`tests/test_hand_comparison.py`. The optional
`tests/test_hand_rotation_gradients.py` checks finite gradients at identity and
bounded proper rotations in a CPU Torch environment; it needs no MANO assets.
Actual hand fitting still requires a separately licensed local MANO model and
its own prefix/full validation. Detector agreement and stereo consistency do
not establish ground-truth hand accuracy or control readiness.

`tests/test_hand_temporal.py` verifies nonuniform-timestamp acceleration, missing
sample boundaries, transaction rejection, geodesic backtracking, rotated image
coordinates, augmentation consensus and the separation of 2D candidates from
metric/control coverage. Visibility tests change every held-out depth pixel to
ensure those pixels cannot affect fitting weights.

`tests/test_video_hand_observations.py` checks crop assignment independent of
MediaPipe, physical hand identity, crop provenance, nonzero frame-window origins,
left-hand principal-point reflection, temporal padding boundaries, independence
of fitting pixels from held-out depth, evaluation image hashes, calibrated
secondary-view projection and exclusion of visual annotations from worker mounts.
These CPU tests do not establish checkpoint quality; the optional HaWoR runtime
requires its own bounded image experiment.

`tests/test_hand_point_tracks.py` checks that held-out detector labels cannot
change tracker seeds, and that occlusion, re-query disagreement, physical hand
identity and duplicate seed times reject unsupported observations. It also
checks partial-landmark masking and the separation of inference/evaluation
mounts. The full point-guided trial requires optional TAPIR/MANO runtimes and
real-image checks; CPU tests alone do not establish anatomical accuracy.

`tests/test_hand_sequence.py` checks derivatives at nonuniform capture times,
component/anchor selection across missing observations, detection of finger
jumps with stationary wrists, protected boundary velocity targets and isolation
of sequence-worker mounts. It also checks that rejected context proposals are
not displayed as accepted merely because their baseline frame was accepted.
Optional MANO batching and fitted-pose export require a licensed-model runtime
check; synthetic tests do not validate anatomical reconstruction.

`tests/test_multiview_points.py` checks metric triangulation and extrinsic
direction, sensitivity to baseline length, reciprocal appearance matches,
rejection of repeated/textureless patches, calibration validation and explicit
warnings for nearby fingertip identities. The synthetic translated-texture
case supplies no hand model or predicted depth. These tests verify geometry
and matching logic, not anatomical correctness on real images.
