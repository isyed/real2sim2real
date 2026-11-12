# Separate command motion from tracking, then transfer load to the table

The arm runner can record planned joint motion, feedback-corrected joint motion,
the final kinematic target and the servo command separately. The servo command
contains bias and velocity/inertial feedforward; using its FK as the desired wrist
pose creates a misleading tracking error. `arm_command_observation.capture`
uses scratch kinematics and never writes desired poses into the simulated robot.

Command logging is enabled for table-transfer probes and can be requested with
`record_arm_commands: true`. It remains disabled by default for ordinary runs,
preserving existing record schemas and exact zero-residual comparisons. Enabling
observations on an old baseline does not waive physical state or verdict equality.

Each recorded `arm_command` contains the actual last command time, physical
sample time, planned/feedback/kinematic joint angles in radians, servo controls,
model bias and inertial torques in Nm, velocity feedforward in rad/s, and world
poses of each planned/commanded/actual hand root. The source object reference,
timestamps and scoring anchors are unchanged.

`arm_command_audit --config ...` takes pinned `cases: [{label, result: {path,
sha256}}]`, a fresh `output`, `windows: [{label, start_s, end_s}]`,
`force_start_s`, `force_end_s`, and optional `deadline_s` (at most 300 s).
It requires complete physics with saved qvel and exact command telemetry. The
CPU stage verifies servo reconstruction and command age, then writes summary JSON
and a browser report. Scene/state hashes are checked; force observations are
re-solved in separate data with zero warmstart.

For world positions, the decomposition is exactly
`actual - planned = (commanded - planned) + (actual - commanded)`.
RMS magnitudes and filtered variation cannot be added to assign percentages of
sway: the vector terms are correlated. Rotation errors are evaluated through
relative rotations, not by adding Euler-angle errors. Filtered motion variation
is distinct from tracking error; see [the force audit](GRASP_WRENCH_AUDIT.md).

## Bounded contact-aware transfer

Optional `table_load_transfer: {baseline: {path, sha256}, controller: {...}}`
pins a complete original wrist-feedback comparison. It can change only its
explicit transfer settings, output, purpose, observations and motion condition.
Native gains, torque limits, physics/control clocks, grasp, finger pressure,
object reference and scheduled release remain frozen. Its controller fields are:

| Field | Meaning and permitted range |
|---|---|
| `gain_m_per_n_s` | Upward admittance per excess table force; 0–0.003 m/(N·s); zero disables correction |
| `max_offset_m` | Positive request cap, at most 6 mm |
| `max_speed_m_s` | Positive request slew cap, at most 12 mm/s |
| `confirmation_s` | Sustained-support dwell, 0.02–0.2 s |
| `support_fraction` | Table force threshold relative to weight, 0.25–1 |
| `deadband_fraction` | Force-error deadband relative to weight, 0.02–0.25 |

The existing separate-data wrist observer supplies signed **table-only** vertical
object force at its configured observation ticks. Hand force is not counted as
table support. The controller holds those measurements between observations.
Correction requires the `return` phase and a sustained table load above the
declared fraction of payload weight; airborne carrying and acquisition are
unmodified. It integrates excess table load above weight, with a deadband, into
a common upward wrist request. Loss of support recovers the request toward zero.

Both wrists receive the same applied vertical increment. It uses only the
remaining headroom inside each wrist's **existing total translation norm cap**;
lateral and rotational corrections are retained. Projection feeds back to the
integrator for anti-windup. Request integration is slew bounded; projection onto
changing shared headroom can reduce the applied increment faster. The original
arm joint-speed/margin limits and whole-robot endpoint/midpoint collision guard
remain authoritative for the final command. No larger Cartesian authority is
silently granted.

At the native capture end, a bounded smooth recovery returns the correction to
exactly zero before original finger release starts. This does not open fingers,
change release latching or attach a constraint/force to the object. A stationary
diagnostic without a return phase cannot exercise this transfer feature.
Current residual training cannot stack this controller with an actor; that needs
its own explicit adapter and complete zero-action verification.

## Evaluation and stopping rule

Require exact disabled-policy reproduction, including qpos/qvel/ctrl/timestamps,
original records and verdicts. Verify the unchanged acquisition/airborne prefix
before activation and zero correction before release. Preserve complete rollouts,
native limits and every failed gate. Browser reviews display transfer metadata.

Measure mean/peak table force, signed hand force, premature touchdown relative to
the unchanged object geometry target, wrist tracking, fixed-anchor retention,
clearance, penetration and release. Report the force averaging window and which
samples are supported. Table force above weight is a diagnostic; acceleration
and transient impact also contribute. It is not automatically harmful pressure.
Report loaded table-support coverage after first touchdown as well as conditional
force means: a controller can lower those means while intermittently unloading
the table. The audit exposes this coverage separately from final settle/release
acceptance, using active contact and an explicit 0.1 N threshold.

Lower mean table force cannot promote a candidate that worsens retention or still
fails earlier acquisition/lift gates. A return-only controller cannot repair an
already-failed airborne prefix. Retain such a result as a diagnostic comparison
and redirect the next experiment to the earlier seating, object/wrist offset or
planned command discontinuity identified by the decomposition. Appropriate
complete holds, loaded motion and [all control gates](CONTROL_READINESS.md) still
precede substantial training or a hardware claim.
