# Reusable reconstruction and grasp lessons

The implementation keeps capture-specific data and outcomes downstream. This
index maps findings to reusable mechanisms, checks and explicit remaining work.
It is not an assertion that a new scene, hand or robot has been validated.

[Pad contact planning](PAD_CONTACT_PLANNING.md) adds collision-pad geometry,
orientation, closing paths and static balance. Finite patch samples share one
force budget per pad; failed balance is not a control success.

| Finding | Reusable implementation | Verification / interpretation |
| --- | --- | --- |
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
