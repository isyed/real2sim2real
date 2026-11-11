# Articulated arms with retained grasp hands

An accepted Cartesian-wrist grasp does not establish an arm-executable grasp.
Use separate admission steps: assembled-model checks, coordinated path checks,
bounded actuator-driven physics, and only then policy training. Keep hand joint
commands, source timestamps, task geometry and physical acceptance criteria
explicit so a change of embodiment remains reviewable.

## Reusable components

* `arm_hand_assembly.build_assembly` attaches existing hand subtrees to independently
  placed fixed-base MJCF arms. It removes only explicitly enumerated root fixture
  joints and their actuators. Finger joints, collision geometry, actuator settings
  and task objects remain in the source scene. Assets are absolute and hashed;
  generated scenes and robot meshes belong in downstream artifact storage.
* `extract_mounted_arm` separates an upstream arm and adapter from an included
  hand, recording the original mount transform. It handles MuJoCo's Euler
  convention explicitly; SciPy lowercase `xyz` is not a substitute for MuJoCo
  lowercase `xyz`. Unsupported include/constraint structures fail closed.
* `ur_arm_model.prepare` expands an explicitly supplied official UR description
  using optional Xacro, then imports its collision URDF into MuJoCo. Record and
  justify joint-limit overrides. A vendor parameter file can differ from a newer
  hardware manual; do not silently substitute a nearby arm variant.
* `arm_path_audit` supports one shared base or named independent bases. It checks
  the original wrist poses, relative hand-body FK, joint margins, sampled speed
  limits and nonadjacent arm collisions. Up to eight placements and 1,000 source
  samples are allowed within a required process deadline. When a hand remains a
  fixture, copy its joint axes as well as its root pose and label the partial
  integration. It cannot establish complete two-arm control.
* `arm_replay_pilot` requires a passing two-arm audit covering every archived
  timestamp. It removes both fixtures and drives arm joints with torque-limited
  position actuators, bias compensation and optional velocity feedforward. Finger
  commands replay the archived controller outputs. Only initialization writes the
  free object's pose; no equality, mocap, external force or repeated object-state
  assignment assists the task. Arm speed, joint margin and collision checks supplement
  the existing grasp/support/reference checks. The explicit-PD helper is available
  for diagnostics, but the runner uses MuJoCo implicit actuator damping.
* `arm_replay_review` renders saved physics states on CPU, displaying arm collision
  meshes where no visual meshes exist. It reuses portable WebM/MP4/GIF playback,
  including embedded WebM for restricted IDE previews. It does not rerun dynamics.
* `arm_replay_comparison` verifies saved scene/state hashes and compares reports
  for the same pinned arm audit. It preserves rejections, reports missing reference
  measurements as unavailable, and lists differing controller settings. Its HTML
  links to the individual videos where present; it never reruns physics.

## Live finger pressure on assembled arms

`arm_replay_pilot.finger_control_mode` defaults to `archived`, preserving saved
finger-command transfer. Explicit `live_contact_force` binds the admitted
simultaneous bilateral pad-pressure rule to assembled hands through
`arm_finger_feedback`. Both modes use audited arm paths. Replay and live feedback
are separate experiments, labeled in results and reviews.

The live adapter reconstructs uncorrected pregrasp/open/closed finger waypoints
on the original approach/closing clock. It does not add feedback on top of saved
pressure corrections. Per-pad control responds to actual solved forces and
supported-load estimates. Original support-hand bias compensation still acts
through bounded position actuators. Only existing finger actuators are written;
source plan/settling hashes are recorded. Arm/object states and native limits
remain outside this adapter's ownership.

The binding requires an unchanged admitted simultaneous pad plan and the source
pressure settings. It rejects posture-transition controllers, unplanned closing
variants and a separate finger-trim owner. These need their own adapters and
admission. Simulator forces/acceleration remain privileged observations.

A declared nominal feasibility experiment may transfer a passing nominal grasp
to screened arms while broader payload robustness remains unresolved. Preserve
that scope and all failed robustness cases. Nominal diagnostic success does not
admit global control readiness, hardware execution or substantial policy training.
Repeat the appropriate matrix after selecting an arm/controller setup.

## Separate physics resolution from controller frequency

By default the outer arm/finger controller updates on every physics step.
Changing `timestep_s` changes both numerical integration and feedback frequency.
Identical gains alone do not make this a pure integration study.

Optional `control_period_s` holds the outer command between updates. It must be
an exact integer multiple of the physics step and at most 10 ms. For example,
1 ms control uses four 0.25 ms steps or eight 0.125 ms steps per update. A fixed
controller clock uses the same scheduled reference times and passes the declared
period to pressure filters, rate bounds and tracking compensators. MuJoCo's
implicit motor/contact dynamics still integrate on each physics step.

Results record both periods and update counts. Comparisons expose differences
explicitly. An absent setting preserves the legacy path; verify its state archive
after shared-runner changes. Test convergence using a fixed outer period, then
evaluate controller frequency separately. Do not select only a numerical setting
that happens to pass a narrow gate.

## Configuration boundaries

`assembly.arms` declares the MJCF path, namespace, source hand root, removed fixture
joint names, mount body and rigid `mount_from_hand` transform. Optional adapter
cylinders and visual-mesh collision proxies are assumptions recorded in a receipt.
They are not verified manufacturing geometry. `arms` declares arm joint names,
collision bodies, source/target hand prefixes and IK seeds. `placements[].bases`
maps each independent base body to `position_m` and `yaw_rad` in the existing world.

A dynamic pilot pins the audit hash and passing placement, declares proportional
and damping gains in Nm/rad and Nm·s/rad, and specifies every arm joint's torque
limit. The underlying task's clearance, slip, opposed contact, load-sharing and
penetration gates are reused. The initial reference gates are 20 mm and 10 degrees,
matching the guided-grasp pilot. This is a diagnostic transfer of saved commands,
with optional new object feedback, not a trained policy or a reconstruction of
the original fixture controller.
The old fixture's Cartesian force/torque limits are not equivalent to the new
arm's joint torque limits; do not describe these as identical controllers.

```bash
python -m real2sim2real.controller_actor_training.arm_path_audit --config /private/path/audit.json
python -m real2sim2real.controller_actor_training.arm_replay_pilot --config /private/path/pilot.json
python -m real2sim2real.controller_actor_training.arm_replay_review --directory /private/path/pilot-output
python -m real2sim2real.controller_actor_training.arm_replay_comparison --results /private/path/baseline/result.json /private/path/feedback/result.json --output /private/path/comparison
```

## What an integration result means

Vendor flange compatibility, a demonstrated hardware combination, an exact CAD
assembly, an IK pass and a physical rollout pass are different evidence. Keep
them separate in the report. Independent bases improve placement freedom but
their stands, cables and unobserved environmental obstacles also need modelling
before hardware claims. Sampled collision tests do not certify continuous clearance.

When IK passes but contact fails, inspect actuator saturation, wrist lag and
load sharing before changing hardware or training a policy. Explicit high-gain
velocity torques can oscillate on low-inertia wrists; implicit actuator damping
is a different, separately logged controller. Velocity feedforward is an ablation,
not an assumed improvement. Retain unsuccessful trials and original thresholds.

## Isolate tracking before adding feedback

`contact_condition: "no_hand_object_contact"` disables only hand–object body
pairs, including object descendants. It retains robot mass, finger commands,
gravity, table contacts and robot/environment collisions. The object is free and
usually stays on the table. This condition is always rejected as a task success;
it isolates the additional effect of grasp contact on wrist tracking.

`exclude_adjacent_arm_collisions: true` excludes only neighboring arm bodies
identified through fixed mount links. Every pair and reason is recorded. This
can remove artificial joint interference in imported collision models. Inspect
the model first: it is a documented collision-model correction, not evidence of
hardware clearance. Cross-arm, nonneighbor arm, hand and environment contacts
remain active. The existing nonneighbor collision acceptance limit is unchanged.

The runner reports per-phase Cartesian position/rotation error for each wrist,
per-joint torque/saturation, actual velocity and joint margin. `wrist_tracking`
compares against the frozen audited path; `commanded_wrist_tracking` compares
against the possibly corrected command. This separates intentional target changes
from servo error. Scene and state hashes, excluded pairs and snapshots of the
runner/feedback/diagnostics implementation accompany the result.

Test loaded and isolated conditions with the same gains, clock and torque limits.
Evaluate timestep sensitivity with `timestep_s` before promoting a narrow pass.
Position accuracy alone does not establish grasp retention or reference fidelity.

## Object-relative correction and second-hand load sharing

`arm_object_feedback` computes the wrist pose that preserves the nominal
object-to-wrist transform on the observed object. Translation and rotation
corrections have explicit rates and travel bounds. A damped Jacobian maps this
world-frame correction into arm joint offsets; command speed and joint margins
are bounded before torque-limited actuation. Finger commands remain archived.

`object_feedback` requires `gain_s_inv`, `max_speed_m_s`,
`max_angular_speed_rad_s`, `max_offset_m`, `max_angle_rad`, `minimum_contact_n`,
`maximum_reference_error_m` and `maximum_reference_angle_rad`. The last two are
controller freezes against the source trial's object pose. They do not replace
the independently scored demonstrated-object trajectory. `feedback_phases`
explicitly maps `right`/`left` to active source phases; an initial controlled
comparison can use only `hold` after both hands are seated.

The primary hand requires thumb and opposing-finger contact. The support hand
requires measured object contact. Loss of contact or excessive reference deviation
freezes the correction; it does not chase an arbitrarily displaced object. Optional
`support_load` uses the existing support admittance settings:
`target_weight_fraction`, `gain_m_per_n_s`, `max_speed_m_s`, `max_offset_m`.
It adjusts the support wrist vertically toward a fraction of object weight.
Both its offset and the combined translation are bounded. Counters and offsets
show whether feedback actually engaged: an enabled option with zero updates is
not evidence of force-sharing performance.

The simulation adapter reads privileged poses and solved contact forces. A hardware
implementation needs calibrated pose/wrench estimates and delay/noise validation.
It never writes object state after initialization, adds object forces, relaxes
contact criteria or certifies mounting hardware. Compare baseline, object feedback
and added force sharing under identical conditions; retain a failed variant as
evidence rather than silently replacing the baseline.

## Reduce arm lag without changing the demonstration

`arm_tracking_control.ArmTrackingCompensator` provides causal filtered velocity
feedforward and optional integral compensation of arm joint position errors.
The velocity term has native speed and explicit acceleration bounds. The integral
has offset/rate bounds and freezes outward integration when the actuator reaches
its torque limit. Neither option changes the scored trajectory or its timestamps.
Use `tracking_phases` to restrict activation, for example to `hold` after grasp
establishment. An integral that improves tracking can still damage the grasp;
evaluate support drift independently.

`InertialFeedforward` adds a filtered, capped arm-inertia term, `M_arm(q) qdd_ref`,
to actuator bias compensation. Accelerations are derived from the fixed audited
joint sequence at its native timestamps. The reference is known in advance;
the torque filter is causal. This is an approximate model-based feedforward term,
not full inverse dynamics or MPC: it omits independently commanded finger
accelerations and contact forces. It uses only arm motor commands, retaining
the final actuator torque caps. Object forces and object-state writes are never
used. Assess actual velocity, contact and margin gates, not only command bounds.

The optional `tracking_compensation` settings are `velocity_fraction`,
`filter_time_s`, `max_acceleration_rad_s2`, `integral_gain_s_inv`, `max_offset_rad`
and `max_offset_speed_rad_s`. `inertial_feedforward` requires `time_constant_s`
and `torque_fraction`. The legacy unfiltered `velocity_feedforward` option cannot
be combined with the filtered compensator. Implementation snapshots and peak
inertial torques are included in the saved run.

`arm_reference_evaluation` reuses FlashCHORD object ADD-AUC and MPPE functions
against the immutable grasp-aligned reference. It reports moving-reference and
complete-hold cohorts separately. Scoring verifies hashes, complete records and
source-clock correspondence; it does not shift time to reduce error. MPPE uses
object-local keypoints, not human fingers. These local pilot scores use estimated
object geometry and a reconstructed, grasp-aligned reference; they are not
official challenge scores against held-out ground truth.

```bash
python -m real2sim2real.controller_actor_training.arm_reference_evaluation --directory /private/path/pilot-output
```

Public integration references: [Sharpa flange CAD](https://github.com/sharpa-robotics/sharpa-opencad-library),
[Franka/right-Sharpa assembly](https://tml.stanford.edu/cloak/setup/sharpa.html),
[official UR descriptions](https://github.com/UniversalRobots/Universal_Robots_ROS2_Description).
External models and their notices stay with their downloaded assets; this project's
Apache license does not replace the licenses of those assets.

## Search a whole path before changing the controller

Each audit placement may declare `ik_seed_indices`, mapping arm names to indices
in their explicit `initial_seeds_rad` lists. The selected seed is used only at
the beginning. Subsequent samples continue from the preceding solution; no
mid-motion branch jump is used to bypass a collision. Omitting the mapping
preserves the previous initial multi-seed behavior.

`arm_path_planning.rank_paths` shortlists only paths passing all existing pose,
joint, speed, hand-frame and collision checks. Among passing paths it prefers
lower speed utilization, then larger joint margin. This is a ranking of sampled
paths, not a global planning or continuous-clearance certificate. Sparse discovery
can underestimate peak speed and miss collisions; every archived timestamp must
be re-audited before replay. Symmetric base placements are not mandatory: a
bimanual grasp can require asymmetric arm placement while both hand targets and
the object trajectory remain unchanged. Bases, stands, mounts and cable routing
remain explicit hardware assumptions.

A collision-free path and a passing object-position error are not sufficient for
control acceptance. Longer motion can lose opposing finger contact or exceed
wrist-frame slip while both arms remain clear. Keep these failures independent.

## Diagnose payload and friction failures by phase

`arm_replay_comparison.phase_object_deviation` compares each run with the first
run on matching phases and clocks, without alignment. It reports first/last
world displacement and the maximum object-translation difference per phase.
Missing or mismatched timelines remain unavailable. The first run is a saved
simulation baseline, not a human or ground-truth reference. This distinguishes
errors introduced during support insertion from later arm-path tracking error.

`arm_clearance_audit` reconstructs robot/table intersection witnesses from saved
states using position/collision calculations only. It verifies scene/state/audit
hashes, bounds the archive to 1,000 samples and records per-phase maxima and
geometry identities. It excludes object/table support from robot contacts. It
does not infer contact forces, rerun dynamics or change historical acceptance.

```bash
python -m real2sim2real.controller_actor_training.arm_clearance_audit --directory /data/trial
```

## Experimental contact corrections

These options are opt-in experiments, not default improvements or accepted
replacements for an existing passing controller. Repeat nominal acceptance
before freezing any candidate for a matched robustness cohort.

`finger_contact_trim` integrates a declared finger's normal-force deficit along
explicit signed flexion-joint directions. The adapter permits only owned,
bounded, unit-gear position actuators in the primary hand. Rate, travel, physical
joint/command limits, primary-thumb contact and fixed-reference position error
bound the correction. It does not alter force limits, friction, arms or object
state. Saturated closing travel without recovered opposing force is evidence to
revisit contact placement, not permission to raise gains until acceptance.

The force channels and object pose are privileged simulator observations. A
nonzero normal force is not a friction-cone or force-closure certificate. Record
which individual fingers actually carry opposing force; many articulated target
fingers can still produce only one effective opposing contact.

`precontact_support_alignment: true`, with explicit `object_feedback` and a
support-approach feedback phase, lets retained primary opposed contact enable
the existing bounded support-wrist pose servo before support contact exists.
Seating and hold still require measured support contact. The reported support
force remains the actual measured force; it is never replaced by primary force.

Alternatively, `support_height_alignment` latches one vertical object discrepancy
at support-approach entry and rate-limits a fixed support-wrist offset. It requires
retained primary contact and a bounded initial pose discrepancy. It cannot be
combined with continuous object feedback. Later object observations do not change
the latch. This avoids continuously chasing the object, but **does not establish
insertion clearance**. Even a small downward offset can put the hand into the
table and create large loaded tracking errors. Such a trial must remain rejected.

When payload changes the object's pose before second-hand insertion, replan the
support approach against the finite table and the observed pose range. If access
is insufficient, revisit the preparatory grasp/lift sequence and re-audit its
reference anchoring. Do not treat small online offsets as a substitute for a
collision-checked insertion path or relax contact gates to accept pose tracking.


## Diagnose retained contacts at a declared feedback rate

Optional `contact_patch_motion` adds named pad-force, tangential material-velocity,
rolling-compatible angular-motion and patch-centroid telemetry to live arm runs.
The runner copies saved qpos/qvel/ctrl/act and the solver warmstart into a separate
MuJoCo data object and calls forward dynamics there. Re-solved contacts never
feed the live controller or its warmstart. Saved qvel is included only when this
instrumentation is requested. Verify qpos/ctrl/timestamps against a pinned run;
a different telemetry archive hash alone does not imply changed physics.

The sampled telemetry is not a continuous slip trace. `arm_retention_audit` reuses
acquisition diagnostics, separates early/late hold windows and plots actual object
motion in each measured wrist frame. It retains source verdicts and their original
acceptance anchors. Coherent sampled forces and the preceding outer controller's
force observations remain separately labeled. Centroid migration can include
contact switching or rolling and must not be called material slip.

```bash
python -m real2sim2real.controller_actor_training.arm_retention_audit --results /private/run/result.json --output /private/retention-audit
```

Explicit `finger_feedback_override` admits only normal observation/threshold,
`contact_anchor` and `loaded_tangent_gap_m` changes. It cannot raise force targets,
joint offsets, native actuator limits or acceptance thresholds. Resolved settings
and source plan hashes are recorded. These are experimental controllers, not
promotions of the source controller.

`contact_anchor` specifies `gain_s_inv`, `max_speed_m_s`, `minimum_force_n` and
`maximum_error_m`. After lift, each sufficiently loaded pad can latch one observed
contact point into its body and object frames. A bounded tangential task is solved
jointly with the original pressure task using existing owned finger actuators.
Anchors never relatch to hide slip; missing contact or excess anchor error freezes
the tangent task. Wrist/object state and the demonstrated reference remain outside
its ownership. Planned-normal and solved-normal ablations must be compared
separately: changing both at once cannot isolate an anchor's benefit.

Optional `loaded_tangent_gap_m` continues the original planned-normal pressure
rule past its default tangent guard only while that pad has measured object
contact above `minimum_observed_force_n` (default 0.1 N). The original normal-gap
guard still applies. Unloaded pads retain the source tangent guard. The explicit
loaded guard must be wider than the source guard and at most 30 mm; per-pad
continuation counters show whether it engaged. This is a change to controller
observation gating, not a relaxation of geometry or physical success criteria.
It requires its own matched retention and numerical study.

Keep a failed correction in the record. Neither contact existence, reduced wrist
tracking error nor a smaller numerical timestep by itself admits a grasp. A
candidate must complete the intended reference and terminal hold at a declared
outer update period before a frozen payload/friction matrix or actor training.


## Stationary acquisition diagnostics and admitted finger replanning

`motion_condition: stationary_hold` retains the audited acquisition, lift, sample
clock and complete declared duration. At first hold it fixes arm joint commands
and the independently specified object reference at their first hold values.
`demonstrated` is the unchanged default. Results label the condition and whether
demonstrated motion was actually executed. A stationary pass is not a trajectory
pass; do not pool the two tasks. Both use the same physical retention gates.

An optional `finger_plan` supplies `plan`, `sha256` and `candidate_id` from an
admitted fixed-wrist `bilateral_patch_replan`. `arm_finger_plan` requires matching
source-plan ancestry, original task initialization, both finite pad patches and
shared-budget gravity balance. FK checks all four wrist waypoints; original
finger pregrasp/open commands and hand identity remain fixed. Grasp/closed finger
changes stay within the planner's 0.35 rad radius and native limits. Bindings,
new targets and source hashes are saved separately from the original arm audit.

The assembled-arm runner checks the new nominal finger commands against every
audited arm sample in scratch simulator data, including robot/environment
collisions. It excludes object contacts from this check because the source
object trajectory is not a physical prediction for the new grasp. Static contact
admission and fresh unassisted physics remain separate gates. Feedback offsets
are also not predicted by this kinematic screen and remain subject to actual
rollout collision, retention, torque and speed checks.

A new finger plan can enter demonstrated motion only with `validated_stationary`
(`result`, `sha256`) pointing to a complete passing stationary arm run. Promotion
pins physical artifacts and preserves finger plan, arm audit, controller clock,
gains, force settings, geometry and native limits. Only the named motion condition,
output/prose and bounded stage deadline may differ. Reference timing and final
hold duration are retained. The existing archived/default route remains available
with its original behavior.

## Choosing update rates for an embodiment

A 1 kHz outer controller is a declared simulation diagnostic, not a universal
hardware requirement. The [UR7e manufacturer specification](https://www.universal-robots.com/manuals/EN/HTML/SW5_22/Content/prod-usr-man/complianceUR7e/H_g5_sections/appendix_g5/tech_spec_data_7e.htm)
lists a 500 Hz system update frequency; the
[official UR communication guide](https://www.universal-robots.com/developer/communication-protocol/)
lists ROS/ROS2 and real-time interfaces at 500 Hz. Communication frequency does
not certify end-to-end tracking, and the hand's command/tactile rates must be
verified separately. A common 500 Hz simulator check is a sensitivity experiment,
not a verified model of all vendor inner loops or tactile sensing.

Use independent fixed physics and feedback clocks. Preserve the earlier rate as
a matched diagnostic cohort, explicitly declare new rates, and validate moving
contacts as well as stationary holds. Similar stationary performance does not
establish moving-task rate robustness. Never choose a timestep or feedback rate
only because it crosses a narrow success gate.
