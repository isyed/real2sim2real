# Continuous object-reference acquisition and motion

`controller_actor_training.continuous_reference` creates a kinematic command plan
on the capture clock. Robot approach and closure use the video's pre-lift interval.
The demonstrated object motion supplies lift and return. There is no separate
robot lift or restart of reference frame zero after acquisition.

The input is a hash-pinned live-finger arm baseline with admitted grasp geometry,
scene and object reference. A rejected motion baseline can supply geometry and
controller settings; its rejection remains. Output schema
`real2sim2real.continuous-reference-plan/1` has `physics_run: false` and
`success: false`. States are planning proxies, requiring a fresh dense arm audit
and actual free-object dynamics.

## Planning contract

Captured poses, frame IDs and timestamps remain pinned. Pose increments use the
capture-world axes and an explicit resting-object anchor, which can be the existing
gravity-settled task pose. This does not reproduce the capture's absolute starting
pose. `continuous_protocol` defines the new timing; donor reference settings
identify the source archive.

The reference runs from time zero. Original approach/closing finger waypoints
finish on that clock. Planning rejects translation exceeding 10 mm before
acquisition completion. Both wrists follow common object increments while
preserving their admitted relative contact arrangement and native limits.

If reconstruction noise puts the raw reference mesh below the table, planning
can raise its **command** within a declared budget of at most 10 mm. Each adjustment
is recorded. Scoring still uses the raw reference; no mesh, tabletop, capture pose
or threshold is edited. Exceeding the budget rejects planning.

Command sampling is 10–25 ms, within a 1,000-sample audit budget, with subsequent
controller interpolation. Native capture timing has no speed scaling. Terminal
settling adds a declared 1–2 seconds. Optional finger release follows that window;
arm withdrawal is a separate planning task.

```bash
python -m real2sim2real.controller_actor_training.continuous_reference \
  --config /data/continuous-plan.json
```

Required keys: `baseline_result`, `baseline_sha256`, new `output`,
`sample_period_s`, `terminal_settle_s`, `maximum_table_adjustment_m`, and
`deadline_s` (<=900). The bounded CPU CLI writes `protocol.json` and
`source-trial/{result.json,states.npz}`, plus an implementation snapshot.

Optional `finger_release` contains `duration_s` (0.5–2), `settle_s` (0.5–2),
`support_dwell_s` (0.1–0.5), `maximum_clearance_m` (<=0.005) and
`maximum_speed_m_s` (<=0.02). Its start is derived from capture duration plus
terminal settling. The native capture is not stretched. The plan opens owned
finger joints while keeping the wrists at their final object-relative poses.

The live controller additionally requires consecutive observed table support:
table contact, at least half the object weight in table normal force, clearance
between -3 mm and the configured maximum, and translational speed below the
configured bound. It latches once, blends its last pressure-corrected finger
command to the open posture, and disables pressure correction. Delayed or blocked
release cannot silently extend the protocol or become a success. Simulator state
and contact forces are privileged observations, not a hardware estimator.

An acquisition `q_open` posture may still pinch the object at the final wrist
poses. `finger_release_plan` can search explicitly named finger joints against
named hand collision geometries and the object's collision pieces at a pinned
returned state. It preserves wrist/object coordinates and native joint limits,
with at most 0.5 rad change from the nominal open posture. Config keys are
`source_result`, `source_sha256`, `time_s`, `joints`, `geoms`,
`maximum_joint_change_rad`, `clearance_m` (1–5 mm), `maximum_evaluations` (<=200),
new `output`, and `deadline_s` (<=900).

A geometrically accepted `posture.json` can be supplied to arm replay as
`finger_release_posture: {"path": "...", "sha256": "..."}`. The adapter only changes
the release destination of owned fingers, then screens the full arm/finger path.
An endpoint clearance is a planning hypothesis; it never substitutes for the
live release checks below. Preserve rejected nominal opening trials as evidence.

Use module CLIs for this experimental preparation route. Pass the new plan and
its source hashes to `arm_path_audit`, retaining original collision/joint/speed
criteria and screening every timestamp. A passing audit can feed
`arm_replay_pilot`, which also checks the assembled finger/arm/environment path.
Planned object states are never written into dynamics after initialization.

## Evaluation contract

Continuous table-to-table and staged suspended-hold tasks have different cohorts.
Preserve both protocols and identify which one each result evaluates.

- `approach` and `close` end at declared acquisition completion.
- `lift` covers the remaining interval before suspension.
- `hold` covers reference mesh clearance above the existing task minimum
  (50 mm downstream). It can contain motion.
- `return` follows suspension; `settle` follows capture completion.
- Optional `release` and `released_settle` follow supported return.

Raw pose error covers acquisition, lift and return under existing 20 mm/10 degree
limits. Clearance, opposing-contact and load-sharing gates apply to the
reference-defined suspended segment. Actual clearance does not choose that segment.

Both wrist-frame drifts anchor at the first actual saved sample at or after
acquisition completion and cover lift, carry, return and settling. Existing 10 mm
limits remain; these measurements are stricter than post-lift anchoring.
Arm and robot constraints retain their original limits.
With release enabled, retention ends at the declared release start. Intentional
opening is assessed separately; closure/lift failures remain unchanged.

New table-return requirements reject sampled mesh/table penetration above 3 mm.
Terminal settling needs table contact in >=80% of final samples, maximum final
clearance <=5 mm and the declared settling duration within 50 ms sampling tolerance.
Release separately requires actual finger opening within 0.1 rad, combined
hand/object normal force <=1 N (including non-pad contacts), >=80% table-contact
samples, <=5 mm object translation drift and a full declared post-opening settling
window within 50 ms sampling tolerance. All these checks use saved physical states.
Neither this test nor a sampled path audit certifies hardware placement.

`stationary_hold` follows acquisition and the demonstrated lift to the first
suspended sample, then freezes wrists and object target for the remaining clock.
It disables release and omits table-return evaluation. Existing staged inputs
retain their old behavior.

`continuous_retention_audit --results ... --output ...` reads pinned arm artifacts
and reports closure, early-lift and suspended windows: loaded-pad fractions,
coherent forces versus requested pressure, planned normal/tangent gaps, offset
saturation, wrist tracking and wrist-relative object offsets. Diagnostic anchors
do not replace acceptance anchors. A planned-target tangent gap is not an
independent measurement of material slip. Existing contact-motion diagnostics
separate sampled sliding from centroid migration.

Its additional `load_readiness` diagnostic reports the first saved confirmation
of a sustained thumb plus two opposing pads per hand at half their requested
normal pressure. The default dwell is 0.1 s with no observation gap above 30 ms.
Both hands must qualify simultaneously for bilateral readiness. Transient touches,
missing samples or zero release targets cannot create a readiness event. This
uses privileged simulated forces and preceding-step pressure requests; it is not
an independent observation, force-closure certificate or replacement acceptance
gate. Earlier readiness can coexist with worse subsequent drift.

For bounded seating experiments, `finger_feedback_override` may change pressure
activation within closure, normal gain and normal speed within the existing
controller validator. It cannot increase force targets, joint-offset budgets or
joint-speed limits. Check stationary behavior before promoting a moving controller;
label any moving run without that prerequisite as a rejected development diagnostic.

Optional `finger_feedback_override.unloaded_acquisition` enables an earlier,
one-shot opposing-pad seek. Its exact fields are `start_fraction` (0 through the
normal pressure activation fraction), `end_after_close_s` (0–1),
`gain_m_per_n_s` (0–0.003, exclusive of zero) and `max_normal_speed_m_s`
(0–0.006, exclusive of zero). The acquisition window is measured from the
original approach/close clock; neither the reference nor the drift anchor moves.
Only non-thumb pads activate early. Before a pad's first solved contact at the
existing observation threshold, it uses the declared seeking rate. First contact
latches permanently, including transient contact: the pad then uses ordinary
pressure rates and cannot restart fast seeking after losing contact. Existing
geometry guards, force goals, per-finger joint-speed and offset budgets remain
in force. Early offsets survive handoff to normal pressure activation. Release
disables seeking. Per-pad diagnostics distinguish `acquisition_contact_seen` from
`acquisition_seeking`; neither means stable load-bearing contact. Keep a complete
matched stationary comparison before promoting changes to demonstrated motion.

`finger_motor_audit --result ... --output ... --deadline-s 60` verifies the pinned
scene/state archives and reconstructs each saved finger position motor's effort
from its command, position, velocity, affine bias and native command/force limits.
It accepts only scalar hinge, unit-gear, fixed-gain position actuators and excludes
the separate `pilot__` arm controller. It reports per-joint tracking error and
sampled effort/command clipping by phase without a contact solve or physics step.
Low torque-limit utilization can coexist with tracking error in a compliant
position actuator; this does not establish why a particular pad is unloaded.
The source verdict stays unchanged, and the output pins its own implementation.

## Opposing-pad seating and coordinated withdrawal

`pad_seating_plan` fits the eight non-thumb pad chains at a pinned, table-supported
acquisition-completion state. Inputs are `source_result`, `source_sha256`, `time_s`,
`maximum_joint_change_rad` (<=0.35), `target_gap_m` (-0.5 to 0 mm),
`maximum_evaluations` (<=150), new `output` and `deadline_s` (<=900). It preserves
wrists, thumbs, object pose and native finger/control limits. Each proposal records
collision witnesses, their agreement with the reconstructed surface and the
original physical verdict. Signed query distances must agree with their witness
segments. Neither a geometry fit nor a witness from simulation is an independent
contact observation.

Optional `nonpad_clearance_m` (0.5–3 mm) also fits clearance from the object for
every collidable non-pad geometry on the controlled finger-joint bodies. This
includes unnamed proximal colliders but excludes visual-only geometry, unrelated
fingers and the designated pad. Position-only contact queries use 30 mm temporary
margins on this scratch model; the margins never enter dynamics. An absent pair
is recorded as a censored 30 mm lower bound, not as a measured zero distance or
a contact witness. Signed contact witnesses preserve pad/object ordering.
Admission requires pad seating and non-pad clearance within their declared
tolerances, consistent witnesses, source-mesh agreement and the original maximum
robot penetration. It remains a static hypothesis at frozen wrists and thumbs;
the compliant robot may reach a different loaded posture.

Arm replay accepts a pinned `finger_seating_plan: {"path": "...", "sha256": "..."}`.
It changes only closed finger targets and the corresponding pad contact frames.
Optional `close_fraction` (0.5–1) finishes commanded finger closure earlier within
the original acquisition interval. The object reference and acquisition drift
anchor keep their declared clock. Original pressure targets, correction budgets
and robot limits are retained. Fresh full-hand path screening and a complete
passing stationary trial are required before promoting this proposal to motion.

Optional plan `wrist_withdrawal` contains `delay_s` (0.25–1), `duration_s` (0.5–1.5)
and `vectors_world_m` for `right` and `left`. Each vector is <=30 mm and has no
downward component. The withdrawal must finish inside the release window; both
wrists retain their orientations. It creates new arm commands, so a fresh dense
arm audit and assembled arm/finger/environment screen are required.

Finger release also supports optional `thumb_duration_s` (0.5 to `duration_s`).
This selects a shorter opening clock for the owned thumb chains. The wrist delay
must allow that clock plus the 50 ms latch tolerance to finish. The same thumb
schedule is used by kinematic preparation, full-hand screening and live control.
If supported release fails to latch within 50 ms of its declared start, the run
aborts before wrist withdrawal. Accelerating thumb opening can collide with still
closed fingers; a clear arm path alone does not admit it.

Path diagnostics identify the worst colliding geometry pair and timestamp. Table
clearance reuses `arm_clearance_audit`'s collision-geometry height envelope, which
extends the tabletop plane beyond its finite footprint. A positive gap guarantees
clearance; a negative gap alone does not prove collision. This avoids inconsistent
zero returns from separated mesh/mesh distance queries. Existing collision and
physical release criteria still determine acceptance. Original rejection receipts
are preserved, and saved-state clearance audits can verify existing runs without
rerunning physics.

Stationary admission uses a half-physics-step tolerance at a declared endpoint
that falls between integration-grid points. It still requires the complete sample
count, matching controller/configuration and a passing physical verdict; this
rounding tolerance cannot admit a shortened hold.

## Joint closure wrists and loaded finger posture

For optimization against actual acquisition and the complete stationary hold,
see [acquisition rollout search](ACQUISITION_ROLLOUT_SEARCH.md). It coordinates
robot wrist/finger closure within the existing interval and preserves the raw
object clock and original acceptance gates.

`closing_grasp_plan --config ...` jointly fits both wrist displacements and all
ten finger chains from a pinned, table-supported acquisition-completion state.
Required settings are `source_result`, `source_sha256`, `time_s`,
`loaded_response_fraction` (0–1), `maximum_joint_change_rad` (<=0.35),
`nonpad_clearance_m` (0.5–3 mm), `target_gap_m` (-0.5–0 mm),
`maximum_evaluations` (<=100), new `output` and `deadline_s` (<=900).
Use an unmodified continuous arm baseline. Source verdicts are retained.

By default, the empirical response is the original nominal closed posture minus
the saved actual finger posture at acquisition. This includes transient tracking
and is not a static compliance model. Optional `response_time_s` instead selects
a suspended, bilaterally loaded observation from a stationary source. Its saved
position-motor command minus actual posture estimates loaded deflection. Existing
support-hand gravity/Coriolis bias is removed before fitting so live compensation
is not counted twice. Record the calibration time, forces and finger speed; a
single observation remains an approximation, not a validated force model.

The position-only fit checks predicted loaded finger poses while bounding the
actual proposed commands within native limits and 0.35 rad of the original closed
targets. Both commands and predicted poses must remain legal. Self/environment
collision constraints apply to both configurations: compliance must not hide a
nominal command that crosses another finger. Arm joints move by
at most 0.03 rad during fitting; resulting wrist translations and rotations must
be within 6 mm and 0.06 rad. Pad seating, non-pad finger clearance, full-robot
environment penetration and conservative table gap are checked independently.
Temporary 30 mm query margins never enter dynamics. Forces do not enter this
geometric objective; complete loaded dynamics must test the approximation.

The proposal uses `real2sim2real.closing-grasp-plan/1`. It pins all finger goals,
contact frames, original nominal anchors and `closing_wrist_offsets`. Pass the
same offsets to continuous planning and the pinned proposal as replay's
`finger_seating_plan`. Runtime rejects a different wrist spec, an incomplete pad
set, changed nominal anchors, or a legacy finger-only proposal on a nonzero new
wrist preload. Finger changes remain owned by the live finger controller.

`closing_wrist_offsets` has exactly `vectors_object_m` and
`rotation_vectors_object_rad`, each with `right` and `left` vectors. Translation
and axis-angle rotation use the current commanded object's axes. Rotation acts
about the wrist origin without an orbit. Smooth preload rises over the existing
close window, stays fixed relative to the moving object through the capture,
then recovers to zero during terminal table settling before supported opening.
Raw reference poses, capture timestamps, acquisition anchor and acceptance gates
stay unchanged. Zero offsets reproduce the original command path exactly.

Every nonzero proposal requires fresh dense whole-arm IK/collision/speed admission
and assembled arm/finger/environment screening, followed by a complete matching
stationary pass before demonstrated motion. A new grasp still needs complete
return/release revalidation even though its nominal preload recovers before
opening. The original release certificate does not transfer to a changed grasp.

## Fitting release over a wrist-rotation path

`wrist_withdrawal.rotation_vectors_world_rad` optionally supplies `right` and
`left` axis-angle vectors of at most 0.2 rad each. Rotation acts about the wrist
origin in world axes. It does not orbit the object. Optional `rotation_delay_s`
(0.1–1) and `rotation_duration_s` (0.25–1.5) can schedule rotation before outward
translation; both fields are required together and must fit inside release.
Supported-opening admission is checked before the earliest nonzero wrist motion.
The runtime checks one planning interval early (at most 25 ms), because joint
interpolation can start motion on the segment preceding the analytic start time.

`release_path_search` fits selected fingers along this motion. Required config:
`source_result`, `source_sha256`, returned pre-release `time_s`, pinned
`seed_posture`, owned `joints`, `wrist_withdrawal`, `rotation_seeds_world_rad`
(1–9), `clearance_m` (1–5 mm), `fit_samples` (5–17), `verification_samples`
(51–151), `maximum_evaluations` (<=100), new `output`, and `deadline_s` (<=900).
Optional `role` selects `primary` or `secondary`; it defaults to `primary`.
`optimize_rotation` jointly fits a rotation inside a strict 0.2 rad ball and the
finger goal. `refinement_rounds` (0–3) adds up to three worst dense-query timestamps
per round. Joint targets remain within native limits and <=0.5 rad from the original
open posture. The current search requires a common finger-opening clock.
Seeds apply to the selected hand; the other hand's finger goal and wrist rotation
stay fixed while that path is fitted.

The search extracts exact hand geometry onto scratch Cartesian fixtures, verifying
world pose agreement against the saved arm state within 2 micrometres/2 microradians.
Unnamed colliders are matched by body, collision mask and compiled local shape;
ambiguous matches fail. The object stays fixed in these kinematic hypotheses.
Position-only collision queries use a planning margin to detect near contact;
no physics step runs, and forces do not enter the fitting objective. Query margins
are never exported into the dynamics scene.

Primary object clearance progresses from the existing 3 mm contact-penetration
envelope to the requested final separation. `require_bilateral_clearance` additionally
checks support-hand penetration through the path and its final separation, with
the final clearance transition over the last 20% of opening. Full-hand self-collision
and table clearance are checked at every sampled pose. Hand collision geometry
must remain at least 5 mm above the table's conservative height envelope. Historical accepted
postures may initialize a new search; their original wrist contract is retained
as seed provenance and does not admit the new path.

Each candidate exports a pinned `finger-release-posture/1` proposal containing
its exact `wrist_withdrawal` and `release_settings`. Hand-fixture sampling is
explicitly separate from full-arm admission and physical release. The preparation
CLI accepts the same `finger_release_posture` spec as live replay, so archived
finger goals, arm audits and the live owner agree. Planning and runtime reject
mixed posture/wrist/clock contracts. Runtime also requires the exact posture
pinned into the audited source plan.

After fixture admission, run a fresh dense full-arm audit, assembled hand/arm path
screen, and complete free-object dynamics. Preserve rejected candidates and the
source verdict, including when a source has good geometry but fails retention.
Release metrics now report peak/final forces for each hand while keeping the
original combined <=1 N acceptance limit. A primary hand clearing can expose
remaining support-hand contact; individual progress never becomes bilateral or
whole-controller acceptance automatically.

Reference scoring includes acquisition/return and separates `complete_protocol`
from native-duration `moving_reference`; terminal settling cannot inflate the
latter. These use reconstructed references, not official challenge ground truth.

The scalar residual-training adapter rejects continuous results until its
observations, rewards and activation support acquisition anchoring and return.
Staged hold-only semantics cannot silently carry over to this new task.
