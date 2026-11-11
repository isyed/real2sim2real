# Reusable reconstruction and grasp lessons

The implementation keeps capture-specific data and outcomes downstream. This
index maps findings to reusable mechanisms, checks and explicit remaining work.
It is not an assertion that a new scene, hand or robot has been validated.

[Pad contact planning](PAD_CONTACT_PLANNING.md) adds collision-pad geometry,
orientation, closing paths and static balance. Finite patch samples share one
force budget per pad; failed balance is not a control success.

| Finding | Reusable implementation | Verification / interpretation |
| --- | --- | --- |
| A residual policy needs an exact zero-action reference before learning | `arm_residual_env`, shared `arm_replay_pilot.rollout`, `residual_pressure`, `arm_residual_validate` | Bounded pressure-only ownership, separate-data observations, fresh resets, explicit interruptions and exact saved-array/record comparison; reproducing a failed task does not promote it |
| A learned correction cannot silently disappear during later validation | Residual metadata in arm reviews/comparisons and rejection by controller-only stationary/robustness admission | Policy-aware replay is required before transferring a residual-controlled success into another evaluation route |
| A new contact posture can recover one payload and fail another | `grasp_sensitivity` freezes one candidate/controller across the full declared case set | Keep alternate case failures and completion flags; successful cases from different plans cannot be pooled without a separately tested observation-based selection policy |
| A moving contact centroid does not by itself establish rolling or sliding | `contact_patch_motion` records solved-contact point velocities, relative angular motion and separate centroid migration; `contact_patch_audit` requires identical replay states | Common-motion, normal/tangent separation, gap-safe integration and read-only MuJoCo tests; rolling-compatible observations are explicitly not material correspondence |
| Pressure adjustment can leave contact near a curved pad tip | `bilateral_patch_replan` refits bounded pad rays/finger posture with fixed wrists and unchanged object targets | Frozen-joint bounds plus finite-patch, whole-pair path and static-wrench admission; every accepted proposal still requires fresh physical cases |
| Payload changes may need different pressure without independent wrist corrections | `payload_pressure` consumes supported force/acceleration and allocates bounded pressure goals through `BilateralContactForce` | Weight and sharing filters, isolation freezes, slew/force caps and robot-only ownership tests; measured load balance and retained grasp require frozen physical trials |
| A nominal grasp pass can fail after either increasing or decreasing payload | `grasp_sensitivity` pins the complete accepted trial, replays nominal states, then varies only declared mass/inertia and sliding friction | Reject controller/gate overrides and incomplete/changed sources; stop on nominal mismatch; retain physical and reference acceptance separately for every case |
| Improving object orientation can increase sliding within a bilateral grasp | Optional explicit-goal `wrist_coordination` plus `contact_force_control`, admitted through `bilateral_evaluate` | Freeze the stationary goal or reference clock; retain both wrists' relative drift gates alongside world-pose metrics. Pressure, common wrist motion and planned-site tangent correction are separate tested hypotheses |
| Different imitation and task-control strategies need explicit boundaries | `pipeline_profiles`, `object_control_stage`, `control_artifacts` | Typed recipes, pinned configuration, source-preserving adoption and failed-audit admission tests; see [pipelines](PIPELINES.md) |
| A short nominal pass does not establish a longer or robust controller | `reference_extension`, `arm_robustness`, `evidence_assessment` | Preserve reference prefix/clock and frozen controller; reject missing evidence and changed physical gates; keep perturbation failures |
| Rebuilding MuJoCo constants can reset initialized robot state | `arm_robustness.apply_case`, shared `simulation_health` in arm replay | MuJoCo state-preservation regression; nominal archive equality is downstream evidence |
| A requested reference horizon may exceed an old export | `guided_motion.select_interval` | Reject truncated ends, gaps and duplicate frame IDs instead of silently shortening the experiment |
| A moving camera cannot define the simulation world | `egocentric` calibration, camera motion and upright support; `object_task` metric scene export | Explicit units, source hashes and world transforms; finite observed table extent remains an assumption |
| Crop localization and pose-model improvements must be compared separately | `hand_experiment`, `hand_tracks`, shared-crop HaMeR/HaWoR workers | `test_video_hand_observations.py`; common fitter and matched observations |
| Appearance tracks do not guarantee finger identity | `hand_point_tracks`, `hand_observation_policy`, `multiview_points`, `hand_anchor_audit` | Point-track and observation-policy tests; ambiguity/unknown evidence remains explicit |
| Wrist depth crops can include watch/forearm outside MANO | `hand_depth_support` prepares image-defined support before fitting; `hand_support_sensitivity` audits margins | `test_hand_depth_support.py`; preserve fit/reserved sample separation and missing boundaries |
| Smoothing or fallback coverage cannot certify corrected motion | `hand_sequence`, `point_hand_sequence`, `interaction_readiness` | Sequence and readiness tests preserve rejected components and distinguish reference admission from control validation |
| Human fingers can remain uncertain while robot task planning proceeds | `object_task`, `retargeting.object_grasp`, `grasp_kinematics` | Width/height/joint/contact checks are mandatory; optional cue influence is bounded |
| Outermost rays can miss vessel walls; reversing a wrist must exchange contact sides | `retargeting.contact_layout` adds local solid spans, explicit thumb-side targets and gravity lever diagnostics | `test_contact_layout.py` checks hollow geometry, reversed and legacy targets, missing surfaces and invalid inputs; lever ranking is not force closure |
| Collision simplification can remove usable geometry | `object_grasp_scene` checks closure and sampled convex-proxy errors, retaining source visual geometry and inertia | An approximation audit is not a topology certificate; robot/collision assets remain external |
| A reconstructed pose may not be a stable physical start | Explicit reconstruction-preserving or gravity-settled task initialization | Displacement/rest checks retain the rejected pose and label a new task start; no silent reference repair |
| A geometric contact plan is not a successful grasp | Separate contact IK, approach search, free-object simulation and hold evaluation | `test_object_grasp.py`; reject object actuators, welds, gravity assistance and stale inputs |
| Five target fingertips do not imply five loaded fingers | `grasp_hold.contact_support` reports actual per-group normal force and active sample fraction, including palm/other geometry | `test_grasp_hold.py`; contact evidence is not a force-closure proof |
| An underbody hand must have insertion space and transfer real load | `multi_hand_scene`, `support_planning`, `support_control`, `bimanual_grasp` add staged articulated support after the initial lift | `test_bimanual_support.py`; named state/fixture isolation, signed object force, bounded load feedback and parked-hand comparisons |
| Geometric clearance can disappear under actuator tracking error | Optional support-hand bias compensation acts through existing force-limited position actuators | Gravity tracking and free-object regression tests; no direct object or applied-force assistance |
| A force solve is unnecessary for rejected geometric hypotheses | Contact IK uses `mj_fwdPosition` for positions, Jacobians and collision witnesses | Real dynamics remains a separate bounded stage; invalid intermediate geometry is not a physical trial |
| Wrist rotation can be confused with slip | `grasp_feedback.relative_translation_drift` uses the moving wrist frame; `object_grasp_audit` rescores retained states | Rigid-rotation invariance tests; original world-vector metric and historical verdicts remain available |
| Independent wrist corrections can disturb a two-hand grasp | `wrist_coordination` applies one bounded rigid correction about a fixed object pivot and projects it into both fixtures' limits | Relative-transform, optional MuJoCo mapping and no-object-state-write tests; joint feasibility alone does not preserve loaded contacts |
| Initial wrist posture can spend travel needed during lifting | `grasp_posture_search` compares surface-supported finger layouts and restricted IK ranges against both retained controllers | Narrowed-bound/ownership and signed-margin tests; bases, physical limits and acceptance gates stay fixed |
| The closing command can move fingertips far from accurately fitted targets | `finger_closure` audits the unconstrained closed command, supports named offsets and bounded object-local finger feedback | Scratch-state, ownership, force/speed/travel and frozen-controller tests; actual loaded contacts remain the evidence |
| Solver resets can leave finite coordinates and misleading partial metrics | `simulation_health` rejects clock resets and solver warnings before promotion | Clock/warning regression tests; historical reset traces remain unchanged and can be replayed with an explicit rejection |
| Low world drift can coexist with poor grasp retention | Independent world-motion audit plus unchanged opposed-contact and both wrist-frame drift gates | A coordinated candidate cannot be promoted merely for reducing lateral motion; saturation and rejected trials remain explicit |
| Brief lift success can hide later drift or changed support | `grasp_hold.measure_hold` / `hold_rejections` inspect all retained hold samples; `grasp_robustness` separates development and longer frozen checks | Later-sample failure, nonfinite evidence, missing hold and independent contact/penetration gates have regression tests |
| More closure can cause penetration instead of improving grasp quality | Bounded `contact_closure` and `orientation_correction` under unchanged joint/force limits | Feedback tests verify direction, speed/travel bounds and singularity behavior; failed trials remain failed |
| A diagnostic follow-up cannot erase development failure | Pinned comparison/protocol selection in `grasp_robustness.validate_fixed` | `validation_decision` tests prevent later passes from silently promoting a rejected controller |
| Frozen-grasp validation must also freeze feedback and motion | `object_grasp_pilot.verify_frozen_controller` rejects changed gains and, with configuration receipts, changed trajectories or shorter holds | Tests allow longer duration while rejecting retuning; historical outcomes remain intact |
| An FFmpeg decode is not a playback check in an IDE | `review_media` supplies embedded primary WebM and an image fallback; `review_server` supplies loopback HTTP with byte ranges | Conversion/cache/HTML tests, range/path tests, plus actual browser playback and seeking checks |

## Applying the modules to another case

Keep a new capture's calibration, assets, configs and experiment results outside
the public package. Prepare a metric scene with explicit frame conventions and
finite support. Review collision approximation and settling before robot contact
search. For a different hand, review joint/site ordering, geometry-to-finger names,
posture convention, actuator limits and feasible grasp span. The current wrist
fixture does not establish an arm's reachability.

Use the same observations and fitter when comparing localization/pose models.
Keep unknown hand identities unknown; reserve evaluation observations before
fitting. In object-centric mode, declare the object task independently of human
articulation. Evaluate actual simulated support, minimum object clearance,
wrist-relative drift, object orientation and penetration. Freeze selection before
longer or altered-parameter trials, and retain the original outcomes.

Report the extent of validation: observed sample intervals, nominal material
assumptions, fixed-grasp parameter cases, and the sensing available to feedback.
CPU tests, deterministic repeats and saved-state renders serve different purposes;
none alone establish independent reconstruction accuracy or hardware success.

## Findings that require new work

The need for new evidence or a new robot capability is recorded explicitly rather
than represented as implemented functionality. Independently identified human
joint/contact observations still need acquisition for faithful imitation. The
[staged second-hand support adapter](BIMANUAL_SUPPORT.md) now supplies a limited
underbody contact search and measured load sharing. The optional
[common wrist controller](WRIST_COORDINATION.md) adds bounded pose regulation,
with explicit joint-limit and contact failure diagnostics. General coupled bimanual
motion planning, arm integration and a task-specific learning environment remain
unimplemented. Exact simulator pose/contact
feedback remains privileged until a deployable observation path is supplied.
The existing training packages do not automatically make this new task ready for
policy training.

See [object-centric grasp workflow](OBJECT_CENTRIC_GRASPS.md) and
[test instructions](TESTING.md) for interfaces and validation commands.
The [task-first roadmap](OBJECT_CENTRIC_ROADMAP.md) separates physical success,
optional skill-graph execution and later human-derived object-motion accuracy.

## Contact evidence and paired support

An accurate whole-object collision proxy can still misrepresent a thin contact
wall. Check the local union boundary against wall thickness, then check the contact
region against visible image/depth evidence. Missing visibility is not agreement.
Depth-window sensitivity is essential when a wall occupies only a few pixels.

Opposite-side contact sets can balance a gravity moment that defeats a single-hand
grasp. That result does not validate simultaneous hand placement or establish that
the reconstructed contact surface is observed. Keep static balance, observation
support, paired robot IK/collisions and dynamic success as separate stages. See
[Contact region audits](CONTACT_REGION_AUDIT.md) for the reusable implementations.

Before correcting a thin contact surface, distinguish ambiguous stereo from a
repeatable metric disagreement. A smaller image patch can remove adjacent pixels
while also destroying match uniqueness. Direct stereo checks, reverse matching
and textured-image controls are implemented in
[Stereo contact evidence](STEREO_CONTACT_EVIDENCE.md). Successful background
matches do not validate contact surfaces; absent contact matches cannot justify
fitting geometry toward a controller's acceptance threshold.

Better observation does not necessarily produce a useful contact: a visible upper
surface can only push the object down under its allowed friction cone. Search
surface coverage first, freeze an observation-supported shortlist, and test force
directions and gravity balance before spending time fitting hands. See
[Contact site search](CONTACT_SITE_SEARCH.md). Preserve failures and keep point
depth support separate from verified surface normals and finite contact area.

A preserved handle opening does not guarantee hook access: the finite tabletop
can block the turn beneath a rim. Audit the opening, proxy intrusion and the full
tool sweep separately. A required extra gap identifies a pre-grasp tilt/lift
subproblem; changing IK or MPC does not supply that gap. See
[Handle insertion](HANDLE_INSERTION.md) for the geometry checks and module map.

Demonstrated object motion can improve trajectory following while exceeding a
grasp-slip limit. Keep motion fidelity and physical retention as independent
requirements. Transfer from an achieved grasp is not replay of a reconstructed
floating start. Review actual human contact strategy before selecting insertion
templates, and compare the same object-local surfaces before/after settling.
See [Reference-guided motion and strategy](GUIDED_MOTION.md).

Bounded wrist correction can pass a nominal retention test while saturating and
failing a longer hold. Check actual loaded fingers as well as wrist-relative
displacement and reference fidelity; following an object does not prove a stable
contact layout. Freeze candidates before altered-duration/material cases. See
[Grasp retention](GRASP_RETENTION.md).

A settled pose can satisfy gravity while fitting observed surfaces worse than the
original reconstruction. Compare both on identical image rays, report coverage,
and inspect nearby background depth before changing the table or object. Raised
visible upper surfaces do not prove usable underside support. See
[Visible-region geometry](VISIBLE_REGION_GEOMETRY.md).

Independent wrist fixtures can select a grasp that forces the full robot's arms
to cross or exceed wrist limits. Check the mounted hand frames, both arms,
parked/support insertion phases, native joint-speed limits and elbow clearance
together. Individual-arm reachability is not a bimanual path. Preserve source
motion while making robot grasp selection arm-aware. See
[Arm path audit](ARM_PATH_AUDIT.md).
## Arm assembly and two-arm transfer

Keep mount documentation, exact assembly geometry, coordinated IK and dynamic
contact retention as separate evidence. Reparenting a retained fixture requires
copying its joint axes as well as its base transform. MuJoCo and SciPy lowercase
Euler conventions differ. Fixed-base arm placement can admit a path that a shared
torso cannot reproduce, but an IK pass does not validate actuator-driven tracking.
Use implicit actuator damping and capped arm forces in a bounded transfer pilot;
velocity feedforward must be evaluated rather than assumed beneficial. Reuse
unchanged grasp/support gates and keep the free object unassisted. Details and
entry points are in [ARM_HAND_INTEGRATION.md](ARM_HAND_INTEGRATION.md).

Isolate wrist tracking with and without hand–object contact before blaming a
grasp or changing embodiment. Imported neighboring collision meshes can jam a
joint even when a path audit excludes those pairs from its reported collision
score. Record explicit exclusions and retain all nonneighbor/environment checks.
Report nominal-path and commanded-pose errors separately, and compare simulation
timesteps before accepting a near-threshold result. Object-relative correction
during acquisition can follow object lag downward; compare correction after grasp
establishment, and record actual feedback engagement. A configured load-sharing
option that never updates is not an effective load-sharing experiment. Keep the
demonstrated motion, contact gates and torque budgets fixed throughout these tests.

A transient tracking error can come from acceleration demands rather than a
constant offset. Bounded filtered inertial feedforward can reduce that error
without increasing torque limits or retiming the motion. Integral correction
must earn its place through grasp checks: lower tracking error can coexist with
worse second-hand drift. Preserve a frozen baseline, reject unsuccessful
ablations, repeat the candidate and test a finer integration step. A short pilot
pass does not establish full-sequence or material/observation robustness.

Object-trajectory fidelity and human-hand imitation are separate objectives.
Use demonstrated object motion as the task target, robot-specific planning for
feasible grasps, and uncertain human contact observations as cues. Object ADD-AUC
and MPPE can improve without finger imitation; measure them on an explicitly
declared reference, geometry and time interval. Report moving intervals separately
from stationary padding. In lifted-object observation audits, an image-context
region labelled `nearby_table` is not necessarily the tabletop: verify its actual
contents before interpreting depth as table height.

Recent arm/contact planning extensions:

- `arm_path_planning` preserves explicit whole-path seed choices and ranks only
  passing paths. Dense replay admission still applies; coarse sampled speed can
  miss a limit violation (`test_arm_path_planning.py`).
- `phase_object_deviation` locates divergence during insertion rather than
  attributing every later object error to arm lag (`test_arm_replay_comparison.py`).
- `finger_contact_trim` isolates bounded owned-finger recovery with contact,
  reference, rate and travel gates (`test_finger_contact_trim.py`). A failed
  force-recovery candidate remains a diagnostic, not a promoted controller.
- `SupportHeightAlignment` latches a single precontact observation rather than
  chasing subsequent motion. `arm_clearance_audit` makes table intersections
  explicit; neither a small correction nor an IK pass establishes insertion
  access (`test_arm_object_feedback.py`, `test_arm_clearance_audit.py`).

## Separate insertion clearance from contact retention

Raise the support palm around its intended contact region while preserving the
world insertion direction. Audit positive table clearance and the sweep beneath
the object separately; a collision-free final pose does not make its approach
safe. If payload sag is plausible, add insertion space below the object and seat
upward only after insertion. Keep such geometry assumptions separate from fresh
payload physics. See [contact replanning](CONTACT_REPLANNING.md).

Use actual pad points, surface normals and individual finger loads to assess
opposition. Tiny fingertip errors or improved aggregate contact fractions can
coexist with an effectively single-finger grasp. A fixed-wrist finger fit that
hits its bounds with large normal errors calls for a different wrist/contact
layout, not relaxed acceptance criteria.

## Preserve the active pad patch and all bearing contacts

A center ray through a curved pad may select a different patch from the one
supporting a loaded object. Preserve an explicit, collision-hull-validated local
point and normal when a simulated contact is used to initialize replanning.
Label its provenance; it is not new camera evidence. Include any palm bearing
needed for torque balance when jointly changing the wrist and fingers.

Evaluate a loaded fit again from the original task start: rigid object-relative
transfer, opening, approach, preload, force balance and finite patch support can
all reject an otherwise convincing local fit. Small pose residuals do not imply
that every intended pad actually touches the object. Planning failures do not
replace an already passing physical baseline.

## Bilateral grasp admission requires retention under load

Opposing rim contacts can substantially shorten the combined contact lever and
pass static gravity balance while still slipping during a real simulated lift.
Verify both hands together; separately valid hand fits can have intersecting
palms. Explicit hand-axis and inner/outer-wall conventions matter for wrist
initialization. Fixture re-centering can recover planning travel without moving
waypoints, but requires a fresh interpolation/collision check.

Use a stationary hold to distinguish acquisition/retention problems from moving
reference tracking. Label that diagnostic separately so it cannot qualify the
original task. A brief lift or a static force budget is not evidence of sustained
contact-force realization. Independent wrist corrections inherited from a staged
support controller may need a coordinated controller for opposing grasps.

## Assign one owner to the lift offset during reference following

Once a grasp has achieved its initial lift, continuing to regulate the lowest
object vertex can raise the object center as the object tilts. This may conflict
with the demonstrated center trajectory even while the grasp remains loaded.
An explicit frozen hold offset keeps reference motion in charge; evaluate it
under the same physical and pose criteria as the unfrozen controller.

Contact-force feedback can stabilize a complete stationary hold while moving
reference tests still fail grasp drift or rotation. Freeze the grasp, gains,
physical settings and fidelity limits before promotion. Distinguish measured
current contact normals from planned contact labels, and keep per-pad failures
visible beneath aggregate contact/load statistics.

## Validate posture changes during acquisition, not just their endpoints

Different fixed finger postures can pass different payloads. Do not pool their
successful cases into one robustness claim. A supported-load-driven transition
needs its own frozen policy, bounded owned-joint changes, intermediate collision
checks and fresh acquisition/hold/reference tests across the complete matrix.
Keep simulator force/acceleration observations explicit; neither case identity
nor actual simulator mass should select the posture. Retain pressure limits,
object targets, grasp anchors, reference timing and acceptance criteria. See
[supported-load posture transitions](PAYLOAD_POSTURE.md).

Matching a final finger posture and estimated payload does not reproduce the
contact state reached by using that posture from acquisition. Compare pad seating
times, other-link loads and object-local contact locations before attributing
drift to force-budget exhaustion. Similar unsaturated pressure offsets can coexist
with different bearing contacts. Integrate sliding only over loaded observations:
less accumulated slip during missing contact is not better retention. Keep
post-step contact forces separate from preceding-step servo observations. See
[acquisition diagnostics](ACQUISITION_DIAGNOSTICS.md).

## Arm transfer needs both live command ownership and whole-path speed checks

Archived pressure commands do not reproduce a feedback controller under new arm
dynamics. Reconstruct its nominal finger schedule before adding live pressure
corrections; otherwise saved corrections are applied twice. Keep motor, arm and
object ownership separate and pin the source contact plan.

A path can fit poses with clear geometry and generous joint-limit margins while
crossing a wrist singularity. Check native-time joint speeds across the complete
path. Compare continuous starting branches and explicit base placements without
mid-path jumps, retiming or collision exclusions. A longer hold/reference horizon
is a new cohort; retain narrow failures and check numerical sensitivity.

Closer-to-COM contact hypotheses can improve point observation coverage yet
still fail wrench balance. Upward-force capacity alone ignores lateral forces
and moments. Test explicit COM uncertainty with the same physical budgets, keep
normals/finite-patch/access uncertainty visible, and preserve rejected plans.

Physics-step refinement also changes feedback frequency when a controller updates
on every step. Record both clocks and hold the outer period fixed for an isolated
integration study. Similar gains, video speed and target poses do not establish
identical discrete control. Retain numerical failures rather than promoting a
single passing setting. Including all capture object poses after an independent
grasp is distinct from executing the original acquisition and release.


### Contact retention and finite surface evidence

Keep outer feedback frequency fixed when comparing contact integration settings.
Record coherent contact motion on separate simulator data so instrumentation does
not perturb the controller. A pad may retain contact while sliding; a controller
may freeze because its planned material point differs from that solved contact.
Check both before raising pressure. Observed-normal or tangential-anchor changes
can worsen retention and object rotation; evaluate them separately and preserve
rejections. A loaded-pad continuation guard must retain normal distance and
native force/joint bounds, and must not chase an unloaded pad.

Stereo support at isolated contact points does not verify finite pad coverage or
surface normals. Test local observed metric neighborhoods and repeated-view
consistency. Reject noisy/partial patches without replacing the input geometry.
Original reconstructed point height, a settled model height and verified insertion
clearance are different quantities. Additional views or measurements are justified
when the existing calibrated streams cannot resolve thickness or a bearing patch.


### Posture admission and stationary-to-motion transfer

Preserve the intended surface side when projecting contact proposals on thin
walls. Nearest Euclidean distance alone can swap an inner grasp surface for an
outer one. Keep a bounded original-normal neighborhood, disclose every rejected
projection, and recheck both finite pad patches and force balance.

A new finger posture can retain audited wrist poses, but it still needs a fresh
assembled-arm/finger/environment path check and actual dynamics. Preserve the
original acquisition waypoints and native limits. A complete stationary pass can
coexist with moving-reference failure at exactly the same controller rate. Pin
the stationary candidate, clock and physical artifacts before motion; do not
retune between the two and then describe them as one controller pass.

Simulation controller rate is an experimental setting until matched to each
embodiment's documented interfaces and measured latency/sensing. Separate the
hand's command and tactile rates from the arm's update rate. Similar stationary
results at two rates do not establish moving-contact rate robustness.

### Pressure authority before residual learning

A separate robot lift followed by restarting reference frame zero creates a pause
and an elevated replay anchor that are absent from a continuous demonstration.
The continuous protocol uses the pre-lift capture interval for robot acquisition.
Reference-defined suspension, acquisition-anchored drift and table-return checks
must be explicit; post-lift drift and continuous drift are different cohorts.
Bounded table-aware command projection must leave the raw scored reference intact.
See [continuous object-reference control](CONTINUOUS_OBJECT_REFERENCE.md).

Returning an object while still squeezing it does not demonstrate release. Add
an explicit table-supported opening phase and check actual open posture, residual
hand contact and object stability afterward. End retention at the declared release
start without changing its acquisition anchor. A stationary diagnostic must disable
release instead of opening the hands while the object is suspended.

An open acquisition posture is not necessarily collision-free at the returned
wrist poses. Even a newly optimized opening endpoint with positive geometric
clearance can retain thumb contact in dynamics. Verify the opening path, actual
joint response and residual contact through the full settling window. Longer
opening does not substitute for a table-clear unhooking/withdrawal path.

Thumb-first opening can collide with a finger that remains closed. Screen the
complete articulated-hand trajectory on the same finger clocks used by live
control, and retain the pair/time witness when it fails. A wrist translation that
frees one hand can leave the other hooked to the object. Check both hands' residual
contact forces, object pose, and tabletop support through the full release window.

Geometrically seated pads can remain unloaded in dynamics because the robot has
not reached those poses. Inspect actual closure response and early load transfer;
a static fit cannot substitute for stationary admission. An earlier finger closing
command must retain the independently declared object clock and drift anchor.

A free opening endpoint can still cross the object wall midway through the motion.
Fit fingers and wrist rotation against contacts along the complete release path,
then add dense verification witnesses back into the fit. A brief rotation before
outward translation can untuck a finger that straight withdrawal leaves hooked.
Keep the exact finger goals and wrist clock pinned together across planning and
runtime. Hand-only geometry admission still requires a full-arm audit and dynamics.

Measure release forces separately for both hands while retaining the combined
acceptance limit. Removing a primary-hand hook can reveal support-pinky contact
that the previous object displacement had concealed. Additional outward travel can
worsen that hook; fit the remaining hand's opening path with the first solution fixed.

Inspect seating before increasing pressure: unloaded pads at closure, millimetre
normal gaps before lift, or wrist tracking active only after lift can explain later
drift. Compare coherent contact force with requested pressure and preserve existing
force, joint and collision budgets when changing approach/closure response rates.

Measure correction authority with small signed pressure changes and thumb/finger
redistribution before training an actor. Higher force targets can change object
rotation, rolling and the other hand's load; they do not guarantee less grasp
drift. A pad's joint-offset bound or geometry freeze can prevent the requested
target from producing the expected contact force. Preserve those guards and
report requested targets separately from solved forces.

Evaluate complete trials against the original task gates, not a learning reward
or a shorter contact segment. A causal drift policy must use the original fixed
grasp anchor; reanchoring after slip hides the error. Selecting directions and
activation thresholds on the same replay is development. Confirming a nominal
result with a deterministic repeat checks reproducibility, not generalization.
