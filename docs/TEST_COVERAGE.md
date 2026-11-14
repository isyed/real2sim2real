# Test coverage catalogue

For installation and runnable checks, see [TESTING](TESTING.md).

`tests/test_loaded_posture.py` verifies multi-snapshot pad alignment, owned-joint
bounds, inherited versus new violations, signed clearance and unchanged input
states without stepping physics. A static fit cannot establish loaded retention.
See [loaded posture fitting](LOADED_POSTURE_FITTING.md).

`modules/agentic-policy/tests/test_author_loop.py` covers persistent measured
history, explicit retain-baseline outcomes and actual pre-step RPC duplicate
rejection. `tests/test_contact_conditioned_posture.py` covers the opt-in
contact-conditioned correction. Full scene-specific zero-gain replays must also
compare positions, velocities, commands and timestamps on nominal and development
cases before a new author campaign.

`test_contact_transport.py` checks loaded-pad-only screening, required acquisition
evidence, preserved obstacle/proximal/carried penetration rejection, and separate
strict-quality warnings versus physical/contact failures. `test_manipulation.py`
also checks preview-state isolation, independent pressure memory, exact subsequent
actuator-loop reproduction and guard rejection before another physics step.
Full task replay and later inversion remain downstream experiments.

`test_mujoco_step_trace.py` verifies bitwise integration-state equality with a
bounded every-step trace, reset-input preservation, wrapper restoration and
gross contact-point detection without treating an infinite plane as bounded.
`modules/agentic-policy/tests/test_transport_screen_audit.py` verifies separate
stationary/carried-object collision hypotheses, preserved world-obstacle checks
and unchanged live integration state. See [stability diagnostics](SIMULATION_STABILITY.md).

`test_loaded_contact_repair.py` checks bounded loaded-snapshot fitting and live
state preservation. `test_arm_pose_screen.py` checks reachable endpoint fitting,
environment collision rejection, rigid-transform validation and scratch-state
isolation. Complete hold acceptance and continuous arm paths remain downstream
evidence; these tests cannot promote them.

`test_grasp_strategy.py` checks the fresh graph's authority, immutable parameter
copies, once-only plan/hold ordering, rejection before physics, declared hold
duration and missing-feedback semantics. Optional runtime cases execute the
actual GaP scheduler and detect physical-property changes during scene extension.
Feedback/review loading checks both source hashes, sibling ownership and
agreement between the fitted plan and its evaluation before exporting results.
Campaign results, source-scene preservation, full holds and arm feasibility remain
separate downstream evidence.

`test_passthrough_graph.py` checks a single persistent zero-action session,
ordered phase handoffs without added time or steps, rejection of waits/resets/
scripts/parallel routing, nested source-hash validation, truncation and clock
failures. With optional GaP installed, it executes the upstream graph scheduler
against a deterministic test environment. Complete saved-state equality on the
actual arm/hand scene remains separate downstream simulation evidence.

`test_descent_pad_control.py` checks conserved pair pressure redistribution during
slew, signed-load handling, per-pad/rate bounds, exact disabled/prefix goals,
recovery on contact loss and once-only material anchors. Complete zero-policy
state/record equality against original and wrist-feedback controllers and actual
loaded descent/release remain separate downstream evidence.

`test_slip_tilt_feedback.py` checks fixed anchors, correction sign/rate/travel,
coordinated object-pivot tilt, contact/reference freezes, blocked downward slip,
supported-release recovery and stationary offset retention. Its optional runtime
case rejects a collision at a command midpoint even when both endpoints are
clear, while preserving live simulation state. Exact observed zero-gain replay
and complete downstream physics remain required separately.

`test_bounded_grasp_correction.py` checks pinned physical sources, frozen controller
settings, owned finger commands, correction rate/travel limits and stationary
retention of offsets. Its optional MuJoCo case checks object/world frame mapping,
scratch IK state isolation and rejection of coarse waypoint interpolation into
release. Full zero-probe state equality and actual loaded-motion/stationary
acceptance remain separate downstream evidence.

`test_continuous_residual.py` checks actual supported-release stage transitions,
carry retention versus release unloading costs, the fixed acquisition anchor,
complete native-clock admission and pressure decay after carrying. Finger-plan
tests cover diagnostic-only pinned failed holds without weakening normal promotion.
Full staged and continuous zero-action saved-state equality remains a separate
downstream simulation check, including protocol, return and release diagnostics.

`test_continuous_tracking_proxy.py` checks known translation/rotation metric values,
surface-ADD invariance under body-origin changes, input immutability, timestamp
interpolation and rejection of missing coverage. Downstream equality against the
official registered-pose formulas on the same reconstructed inputs validates the
math only; it does not provide independent ground truth or benchmark scores.

`test_release_path_search.py` checks hand-role contact residuals, structural matching
of unnamed colliders, bounded rotation parameters and position-only near-contact
queries without physics advancement. Wrist withdrawal tests check world-axis
rotation about the wrist origin, separate rotation/translation clocks and support
admission before the earliest motion. Opening-posture tests reject mixed wrist/clock
contracts and protect fixture/arm/object coordinates during planning. Continuous
release tests retain the combined force limit despite per-hand diagnostics.

`test_pad_seating_plan.py` checks bounded, pinned opposing-finger changes, contact
frame validation, atomic rejection and preserved acquisition timing. Withdrawal
tests check translation/duration bounds and supported-unhooking admission.
Mounted-hand and finger-path tests verify independent thumb clocks and conservative
table clearance without writes to the physical state. Stationary promotion tests
accept integration-grid endpoint rounding and reject a genuinely shortened trial.

`test_finger_release.py` checks table-load gating, smooth opening without a command
jump, blocked release and no automatic re-close. `test_finger_release_plan.py`
checks pinned, bounded finger-only posture binding without changing acquisition.
Mounted-hand runtime tests verify release does not write arm or object state.
These unit checks do not establish a successful physical release; saved-state
opening, residual-contact and tabletop stability checks remain required.

`test_continuous_reference.py` checks capture-clock motion without an independent
lift, raw-reference preservation under bounded table command projection, nonfinite
input rejection, reference-defined phases, acquisition-anchored drift and table
return. Reference-scoring tests ensure acquisition/return are included and terminal
settling is excluded from the native-duration cohort. Downstream dense IK, complete
free-object physics and exact staged zero-action regression remain separate evidence.

`tests/test_pressure_policy.py` checks copied bounded requests, causal drift
activation with a fixed-anchor flag, saturation, invalid settings and rejection
of changed gates, clocks or incomplete timelines in pressure-authority trials.

`tests/test_arm_residual_env.py` covers bounded pressure actions, policy clocks,
immutable source binding, reset/terminal semantics, time-normalized rewards,
fixed grasp anchors and deadline truncation. The extended mounted-hand MuJoCo
fixture in `test_arm_finger_feedback.py` checks pressure-target injection without
arm/state/limit writes and verifies that observation capture preserves the live
solver state. Complete zero-action physical replay is separate downstream evidence.

`tests/test_arm_finger_feedback.py` checks native finger waypoint timing,
reconstruction before feedback, and exclusive actuator ownership. Its optional
MuJoCo test binds hands under actuated arms while preserving physical arm/object
state and joint/force limits. `test_reference_extension.py` checks explicit longer
holds without reference retiming. Regional contact tests check equal sampling,
declared rigid frames, disjoint faces, empty-region rejection and fixed-budget
COM sensitivity. Actual path/dynamics and observation admission remain separate.
Arm-replay tests also verify exact physics-step/controller-period ratios and
matching scheduled update times. Comparison tests require changed feedback
periods to be visible even when the physics step stays fixed. A downstream
default-path state-archive regression remains separate physical evidence.

`tests/test_acquisition_compare.py` checks phase/gap-safe sliding integration,
coherent-force versus preceding-servo observations, world-transform invariance,
missing unloaded centroids, unchanged failed verdicts, explicit configuration
changes, matched clocks/physics/wrists and immutable state-identical audit inputs.
It also checks safe HTML data embedding. These diagnostics do not admit control.

`tests/test_payload_posture.py` checks nominal/heavy deadband, continuous load
interpolation, joint/fraction speed bounds, observation freezes, finger-only
command ownership, return to the original pad point and rejection of failed,
incomplete or nonfinite endpoint evidence. Endpoint geometry and transition
retention still require scene checks and complete downstream physical rollouts.

`test_contact_patch_motion.py` separates normal motion, tangential material slip
and patch-centroid migration; verifies gap-safe integration and common-motion
invariance; and checks optional MuJoCo point-velocity telemetry without physical
state writes. `test_bilateral_patch_replan.py` verifies bounded owned-finger
changes while freezing wrists and unselected joints. Actual surface support,
collision admission and fresh physics remain downstream evidence.

`tests/test_payload_pressure.py` checks supported-load estimation with acceleration
compensation, external-support freezing, opposite-sign pressure allocation,
bounded rate/targets and invalid observations. The contact-force ownership test
also runs with payload adaptation enabled in the optional MuJoCo runtime. It
preserves object state and joint/actuator limits. These checks do not establish
payload robustness; repeat the frozen downstream cases before arm dynamics.

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


`tests/test_surface_patch.py` checks missing/collinear point rejection,
two-dimensional pad-radius coverage, noisy depth, proposed-normal disagreement
and repeated-view counts. A passing sampled plane never certifies thickness,
continuous bearing area or insertion. Real stereo evidence remains downstream.
`tests/test_contact_anchor.py` checks once-only material anchors, tangential-only
correction, rigid-world invariance, speed bounds and contact/error freezes.
The optional MuJoCo force-controller ownership test exercises planned and observed
normal modes with explicit synthetic loaded-pad observations, without modifying
physical object/robot coordinates or native limits. It is an ownership test, not
a simulated grasp-validation result.

Force-control tests also require a wider loaded tangent guard to preserve the
original normal guard and the original unloaded-pad guard, with all native limits
unchanged. Live-binding tests reject force-budget or joint-offset overrides and
preserve source dictionaries. Acquisition tests exercise arm-retention timelines
and freeze counters without changing source records or acceptance anchors.
After shared-runner changes, run an exact saved-state regression downstream:
adding qvel/telemetry changes archive contents but must preserve qpos, ctrl and
native timestamps. Controller changes still require fresh full-length physics.


`test_arm_motion_condition.py` checks that a stationary diagnostic preserves the
source, complete command clock and acquisition prefix while freezing only hold
commands and references. Demonstrated motion is the unchanged default.
`test_arm_finger_plan.py` checks hand identity, original acquisition, bounded
finger changes, FK wrist preservation, immutable stationary physics evidence and
rejection of clock retuning or incomplete holds. Its MuJoCo path-screen test
preserves physical state/native limits and rejects a new environment collision.
Patch-replan tests require thin-wall projection to preserve the intended wall
side and seed offsets to respect frozen joints and search bounds. Reference-score
tests reject stationary runs as demonstrated-motion evidence. Actual candidate
admission, complete stationary holds and matched motion retries remain downstream
physics evidence.

`test_wrench_feasibility.py` checks analytically known pinch loads and motor-limited
capacity, torsional moment transmission, COM moment arms, inner/outer friction
bounds, reference acceleration, servo saturation and a lower-torque-bound case
where maximum load above one does not imply nominal feasibility.
`test_sampled_motion_variation.py` distinguishes smooth cubic trends from actual
sampled variation and rejects irregular clocks or insufficient duration. These
unit checks do not certify retention or attribute visual sway to a control loop.

`test_table_load_transfer.py` checks support/phase gating, confirmation dwell,
common-wrist bounds, conserved lateral/rotational commands, shared-budget
anti-windup and zero correction before release. Optional MuJoCo tests verify
scratch-only command capture and distinguish upward table support from downward
force by other geometry. `test_arm_command_audit.py` verifies exact translation
decomposition and world-translation invariance. Complete downstream disabled/
ordinary replay and pre-activation prefix checks remain necessary after shared
runner changes; passing a table-transfer diagnostic never waives earlier gates.

`test_bounded_policy_graph.py` checks parameter authority, immutable graph routing,
nonfinite/unknown inputs, one native simulation clock, early completion and
exclusive residual ownership. A downstream all-zero proposal must additionally
reproduce the pinned full physical replay exactly before nonzero testing.
It also rejects an abort during the final phase, nonfinite handoff clocks and
invalid deadlines before importing the optional runtime.

`test_arm_motion_comparison.py` checks that a combined video preserves the input
frame count and source files, and rejects mismatched video frame rates rather
than silently retiming one controller.
It checks unequal-length failure cleanup and prevents overwriting source media
or a previous comparison through the simulator-independent export helper.

`test_policy_feedback.py` checks fixed acquisition anchors, release exclusion,
common-motion invariance, gate direction, invalid clocks and missing loaded-pad
evidence. Exporting a real run also verifies reproduced retention metrics and
unchanged input hashes.

Nominal diagnostic helpers: `tests/test_container_colliders.py` checks the empty
container opening, closed bottom and outward signed mesh volume;
`tests/test_diagnostic_verdict.py` checks warning continuation without promoting
control acceptance and latched physical failures. These unit tests do not validate
a scene reconstruction or a loaded robot controller.

`tests/test_joint_interface_collision.py` reproduces an overlapping fixed-base
joint interface, verifies a narrowly scoped exclusion, rejects world/multi-joint
pairs, and confirms that an external obstacle still collides with the arm.

`tests/test_phase_timing.py` checks preserved prefix timing, monotone progression
and inverse clock mapping. Extended `test_pad_pressure_correction.py` checks exact
legacy behavior when options are omitted, bounded tangential correction, owned
joints and atomic rejection of malformed tasks.

Trial boundary and local-geometry checks: `tests/test_joint_boundary_path.py`,
`tests/test_local_plane.py`, and `tests/test_observed_surface_render.py`. The
render check requires the optional MuJoCo/trimesh runtime. See
[repeatable reference trials](REPEATABLE_REFERENCE_TRIALS.md).

Source scene/motion checks: `test_native_tracks.py`, `test_planar_stereo.py`,
`test_bearing_surface.py`, `test_axis_motion.py`, and `test_axis_sequence.py`
cover held-out camera rejection, calibrated stereo ray projection, masked gaps
and metric depth discontinuities, directed-axis inversion, missing-frame clocks,
and recovery of a synthetic moving object without time warping. Run these with
`test_local_plane.py`, `test_container_axis.py`, `test_observed_scene.py`, and
`test_tapered_container.py`. These checks validate software logic, not dataset
geometry or robot readiness. See [source scene/motion](SOURCE_SCENE_MOTION.md).

`test_reference_translation.py` checks bounded translation with unchanged
orientation, exact zero-correction behavior, and smooth phase windows that leave
excluded acquisition/inversion/release commands unchanged. Pair it with
`test_phase_timing.py` and `test_bearing_colliders.py` before loaded scene/motion
comparisons. A downstream saved-state prefix comparison is still required to
detect unintended changes from arm-path resampling or IK.

`test_contact_quality.py` distinguishes individual-pad and simultaneous contact
loss and rejects missing/nonfinite phase evidence. It reports sampled checks,
not robustness probabilities or physical control admission.

`modules/agentic-policy/tests` covers frozen inputs, proposal budgets, explicit
retry behavior, retained nominal success, separate admission criteria, common
artifact export and bundled-source hashes. Its `tools/clean_acceptance.py`
exercises built wheels with all workspace checkouts hidden: automatic/manual/
review modes, a scripted RPC policy and a declared robustness case. The author
is deterministic test code, not an external LLM. `tools/physics_acceptance.py`
compares an independently frozen full-horizon replay with the installed backend.
Neither orchestration success nor an interface replay is a task-policy success.

`test_sampled_reference.py` checks exact accumulated clocks and rejects invalid
or future-rollout arrays in a planned-reference bundle. `test_replay_comparison.py`
rejects command-only differences, nonfinite values, clock changes and mismatched
release/outcome evidence. The installed-wheel `tools/manual_replay_acceptance.py`
runs fresh closed-loop physics for a prepared successful manual baseline, with
expected states outside the simulation sandbox. See
[standalone manual replay](FROZEN_REFERENCE_REPLAY.md).

`modules/agentic-policy/tests/test_skill_catalog.py` verifies all 33 bundled
higher-level skill sources, licenses and hashes, and distinguishes source
references from registered adaptations. `test_manipulation.py` adds optional
MuJoCo checks for acquisition timing/command parity, read-only IK/pad fitting,
carried-object obstacle screening, stale-plan rejection, loaded-hold diagnostics
and support-gated placement/release. Run these with `test_continuous_policy.py`,
`test_placement_outcome.py` and `test_pad_pressure_correction.py`.
The installed-wheel `tools/skill_acceptance.py` exercises the seven adaptations
through isolated graph RPC in a supplied portable robot scene. It intentionally
uses an initial-posture negative control rather than claiming grasp success.

`test_continuous_policy.py` uses optional MuJoCo to verify invalid-command
atomicity, continuous-clock partition parity, nonzero actuator-driven motion,
and final-state capture between sample ticks. `test_placement_outcome.py` checks
independent stable placement and same-home verdicts. These do not establish task
success; downstream GaP integration and loaded trials remain separate.

`test_seating_graph_replay.py` checks the zero-edit seating graph's configure/run
ordering and rejects repeated runs or nonzero edits. Its optional GaP test runs
the installed scheduler and confirms that graph completion cannot turn a failed
physical task into success. Run it in the optional graph environment as well as
the lightweight test environment; downstream full-physics parity is a separate
experiment.

Read-only obstacle/contact evidence and portable comparison media:

```bash
python -m pytest -q tests/test_obstacle_clearance_audit.py tests/test_observed_colliders.py tests/test_review_media.py
```

These cover signed primitive witnesses, nonmutation, malformed mesh-query results,
missing observation semantics, and exact frame-count/rate preservation for a
single-clock comparison video. `review_media.comparison_video` joins matching
saved videos without JavaScript synchronization. Native controls and an embedded
animated fallback work independently of report scripts; IDE playback restrictions
still depend on the preview host.

`tests/test_clearance_envelope.py` checks the whole-object height bound against
rotated vertices with reserved tracking error, including the zero-rotation case
and rejection of invalid geometry/reserves. Endpoint bounds never claim an
accepted arm or insertion path.

Saved-force accounting: `tests/test_traced_contact_forces.py` verifies object-side
force signs, world-frame rotation, explicit surface groups, unknown counterparts,
nonmutation and malformed-frame rejection. It requires NumPy, not a physics rerun.

The environment-observation tests check unconfigured table support, actual object
clearance, finite mesh footprint interpretation, live-state nonmutation and exact
subsequent dynamics. Agentic revision-review tests check missing/stale result
approval and require a new review for each evaluated proposal. A downstream
retained-controller replay should additionally verify full trajectory equality
when adding observation tools to an existing experiment.
