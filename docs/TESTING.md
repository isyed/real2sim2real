# Testing the reusable modules

`tests/test_grasp_sensitivity.py` checks bounded physical cases, mandatory nominal
replay, immutable controller/gate admission, changed/failed/incomplete source
rejection and early stopping on a state-archive mismatch. A trial that never enters
reference motion must still return an explicit completion verdict and fail
reference admission. Downstream free-object cases remain required evidence.

Contact-force and wrist-coordination tests also cover optional tangent/normal
velocity control, joint stops, immutable externally supplied pose goals, bounded
3D corrections, robot-only ownership and rejection of controller retuning between
stationary and moving trials. Actual grasp retention remains a downstream physics
test; improved object orientation alone is not a passing grasp.

`test_control_pipelines.py` validates typed recipes, copied evidence, configuration
hashes, upstream rebinding and rejected grasp/arm admission.
`test_reference_extension.py` preserves native frames, prefix poses, clock and
relative grasp while rejecting truncation and insufficient duration.
`test_arm_robustness.py` freezes controller settings and uses an optional MuJoCo
fixture to check that physical perturbations preserve initialized robot state.
`test_evidence_assessment.py` rejects empty cohorts, missing/false claims,
changed evidence and truthy non-Boolean values. These tests do not establish
physical robustness; downstream matched trials remain required.

`tests/test_arm_tracking_control.py` covers causal velocity/acceleration bounds,
integral rate/travel limits, torque anti-windup and bounded inertial coupling.
`tests/test_arm_reference_evaluation.py` checks that object translation and
rotation affect MPPE/ADD, and rejects empty scoring cohorts. Retention and
reference acceptance still require downstream actuator-driven trials.

`tests/test_arm_object_feedback.py` checks rigid object-relative rotation including
its wrist lever arm, correction rate/travel bounds, contact/reference freezes,
signed support-load response and singular-Jacobian behavior. The optional MuJoCo
contact-isolation regression in `test_arm_replay_pilot.py` checks that table
contacts and nonneighbor arm collisions remain enabled. Real loaded/unloaded
comparisons and timestep sensitivity are downstream experiments, not unit-test
claims of physical success.
`tests/test_arm_replay_comparison.py` preserves missing-reference failures and
hold-only maxima, including transient support-load corrections that return to zero.

`tests/test_arm_hand_assembly.py` checks independent arm placement, retained
finger/contact defaults, fixture removal, free-object preservation, source asset
resolution, MuJoCo Euler mounting semantics and copied fixture axes.
`tests/test_arm_replay_pilot.py` checks torque-bounded control, bias/velocity
feedforward, contact ownership and a MuJoCo force-clipped actuator with an
independently falling object. The actual two-arm path and grasp-transfer results
remain downstream experiment evidence. See [arm integration](ARM_HAND_INTEGRATION.md).

`tests/test_arm_path_audit.py` checks native-clock velocity ratios, clock-reset
rejection, joint-neighbor classification through fixed mounts without hiding
world or folded-arm collisions, and an optional MuJoCo body-pose IK fixture that
leaves the free object's state unchanged. Run this with the existing pad,
guided-motion, grasp, support and wrist tests in the MuJoCo runtime. Actual URDF
mounting comparisons and real-scene arm-path failures remain separate evidence.

`tests/test_grasp_retention.py` checks common-motion invariance, correction
direction, contact/reference freezes, speed/travel limits and robot-only command
ownership. `tests/test_visible_region_geometry.py` checks disjoint image regions,
occluder exclusion, first-visible ray depth and missing-ray preservation. Run the
guided-motion, object-grasp, bimanual-support and wrist-coordination tests in the
optional MuJoCo runtime after changing the shared runner. Baseline state-archive
identity and actual longer/slippery validation are separate experiment evidence.

`tests/test_guided_motion.py` checks timestamp-preserving motion transfer,
feedback coordinates, missing intervals and fidelity independently of stationary
task success. Its optional MuJoCo test checks wrist limits and no object-state
writes. `tests/test_demonstration_strategy.py` checks original starting clearance
and frame IDs without inferring contact. Run the existing grasp, bimanual-support
and wrist-coordination tests after changing the shared runner.

`tests/test_insertion_geometry.py` covers projected openings, swept-probe
clearance, overlapping collision-piece membership and between-sample obstacles.
Run `tests/test_object_grasp.py`, `tests/test_pad_planning.py` and
`tests/test_bimanual_support.py` in the optional MuJoCo runtime after changing
shared prepared-scene verification.

`tests/test_contact_site_search.py` covers observation-first surface sampling,
discovery-only shortlists, lifting-force direction, provenance, deduplication and
exclusion of failed validation from combined contact screening. Existing furniture
support reconstruction remains covered by `tests/test_surface_support.py`.

`tests/test_stereo_contact_evidence.py` checks sparse stereo correspondence and
triangulation against known synthetic disparities, including ambiguous, textureless,
reverse-inconsistent and search-boundary cases. No GPU or private images are needed.

`tests/test_contact_region_audit.py` and `tests/test_paired_contacts.py` cover
visibility/depth evidence, local convex-union boundaries, per-pad force budgets,
paired diagnostic provenance and rejection of unsupported contact regions.
They run on CPU without private observations; they do not certify a grasp.

`tests/test_pad_planning.py` covers pad frames/Jacobians, position-normal fits,
intermediate path collisions, gravity moments, shared pad budgets, nearby opposed
patch evidence and bounded microhole repair. Run it in the optional MuJoCo runtime
as well as the lightweight suite. Synthetic checks do not certify a physical grasp.

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

`tests/test_hand_observation_policy.py` checks symmetric ambiguity weighting,
preserved observations/missing joints, anchor reliability from unambiguous
fitting residuals, weak-anchor correction priors, preservation of the original
image/depth loss normalization, bounded options and fixed image/wrist regions
for post-fit depth diagnostics. The real MANO sequence worker separately checks
that exported retained poses reproduce its retained geometry. Neither test
suite nor regional residuals establish control readiness.

`tests/test_hand_depth_support.py` checks image-only wrist support, withheld and
missing boundary behavior, hash/timeline integrity, fit independence from
reserved/excluded depth, complete-window fitting across missing/time boundaries
and isolation of worker mounts. `tests/test_sequence_readiness.py` ensures that
retained fallback coverage cannot hide rejected components, unknown evidence
blocks reference promotion, requested-frame gaps stay explicit, and a passing
reference never claims controller validation. Boundary uncertainty and actual
finger identity still require capture-specific evaluation.

`tests/test_object_grasp.py` checks surface-width/height constraints, distinct
finger targets, bounded uncertain cue influence, separation of closure/lift
phases and clearance-feedback speed/travel bounds. Optional MuJoCo tests verify
contact Jacobians against finite differences, robot/object joint ownership and
rejection of gravity, weld and object-actuator assistance. Saved-plan tests reject
changed scene receipts, changed settled starts and altered selected grasps or
recorded outcomes. Run this file again
inside the optional grasp runtime to exercise those tests instead of skipping
them. Real-scene settling, collision approximation, nominal lift and fixed-grasp
parameter trials remain separate experiments; a nominal pass is not robustness
or hardware validation.

`tests/test_grasp_feedback.py` checks the direction and bounds of contact and
orientation corrections, wrist-frame drift invariance under rigid rotation, and
the separation between diagnostic validation and development admission.
`tests/test_review_media.py` exercises actual MP4-to-WebM/GIF conversion, complete
frame decoding, unchanged source bytes, cache rejection and HTML fallback
escaping. Browser playback/seek checks remain separate local review evidence.

`tests/test_grasp_hold.py` checks that planned fingertips are not counted
as observed support, later hold failures are retained, nonfinite/missing evidence
cannot pass and contact/penetration gates remain distinct. `test_review_server.py`
checks video byte-range parsing and rejects hidden files, private asset formats
and paths/symlinks escaping the selected review directory. The portable-player
tests also check a self-contained primary video for IDE previews.

`tests/test_contact_layout.py` checks gravity torque against a known lever,
reversed thumb/finger surfaces (including legacy proposals), explicit missing
surface rejection, and local solid spans on a hollow vessel without bridging its
cavity. It rejects open geometry and invalid budgets. Frozen-controller tests
allow a longer hold while rejecting changed gains, approach/lift timing and
shortened holds. Run these alongside `test_object_grasp.py`, `test_grasp_hold.py`
and `test_grasp_feedback.py` in the optional MuJoCo runtime. Geometric tests do
not establish contact stability; saved-state physics trials supply that evidence.

`tests/test_bimanual_support.py` checks namespaced MJCF defaults/references,
bottom-surface ray targets, named state transfer and isolation of each wrist's
axes. Optional MuJoCo tests verify upward contact-force signs for both geometry
orders and robot-only bias compensation under actuator limits while a separate
unactuated object falls. Additional tests reject lateral-force-only support,
measure support-wrist drift, bound load feedback, preserve parked/insert/seat
timing, prevent following a falling object and reject frozen-controller changes.
Use the optional grasp runtime to exercise MuJoCo tests. A matched-duration
parked-hand study and exact single-hand state-archive regression remain separate
experiment evidence.

`tests/test_wrist_coordination.py` checks common rigid motion around an object
pivot, relative wrist-frame preservation, feedback direction/speed/travel bounds,
independent world drift and frozen coordination settings. Its optional MuJoCo
test checks nontrivially oriented wrist fixtures, shared joint-limit projection
and unchanged object/physical coordinates. Run it alongside the grasp/support
tests above in the optional runtime. Passing kinematic checks does not imply
contact retention; compare actual force-limited trials against the pinned baseline.

`test_grasp_posture_search.py` verifies narrowed robot-owned IK bounds without
changes to physical joint ranges, explicit invalid-bound rejection, signed
sampled wrist margins and all-case candidate admission. Run its optional MuJoCo
checks in the grasp runtime; geometric feasibility does not certify loaded contact.

`test_finger_closure.py` verifies named closing-offset ownership and bounds,
frozen settings, scratch command kinematics without physical state writes,
bounded independent finger tracking/force offsets, and rejection of solver clock
resets or warnings. Run it with the grasp/support tests in the optional MuJoCo
runtime. Model contact evidence and unchanged baseline state archives remain
separate experiment checks.

`test_arm_path_planning.py` checks declared seed selection, rejected-path exclusion
and whole-path ranking. `test_finger_contact_trim.py` checks owned hand actuators,
rate/travel bounds, force/reference freezes, physical joint stops and floating
point interpolation at a stop. `test_arm_object_feedback.py` also checks the
precontact force gate and single-observation support-height latch.
`test_arm_clearance_audit.py` checks that saved geometric robot/table intersections
exclude ordinary object/table support. These options require separate nominal
and frozen-condition physics validation; unit tests do not promote them.

`tests/test_contact_replan.py` checks pivot-preserving support rotation, horizontal
insertion and vertical seating, extra opening-gap isolation, bounded finger-only
changes, rotated compiled-mesh clearance, and loaded-wrist rebasing without
moving the object or any world body. MuJoCo-dependent tests need the optional
runtime. Downstream fixture/arm dynamics and matched material tests are required
before promoting any planned contact change; see [contact replanning](CONTACT_REPLANNING.md).

Coupled grasp tests additionally cover rigid loaded-to-initial wrist transfer,
bounded wrist/finger search without changing native limits, explicit off-center
collision-pad patches, and inward-facing opening-wall selection in a declared
world. Optional MuJoCo tests validate pad ownership, local/world transforms and
normal/position Jacobians. Downstream contact rejection and baseline state-hash
regression remain separate evidence from these tests.

`tests/test_bilateral_grasp.py` checks inner/outer rim selection, both-hands-close
before-lift timing, shared vertical feedback, equal namespaced fixture bounds,
mutual-collision two-hand IK without moving the object, and FK-preserving closed
wrist re-centering. It also prevents stationary diagnostics from being promoted
as demonstrated-motion success. Optional MuJoCo tests exercise the actual compiled
hand/fixture semantics; downstream free-object rollouts remain necessary.

`tests/test_contact_force_control.py` checks signed pressure/release, rate and
travel bounds, joint-stop anti-windup, object-specific solved force normals,
finger-only command ownership and unchanged physical state/limits. It rejects
failed/incomplete stationary promotion, changed control gains or relaxed reference
limits, and verifies distinct media names for repeated candidate IDs across studies.
Run the optional MuJoCo cases alongside the bilateral, support, guided-motion and
contact-replan tests after changes to the shared fixture runner.
