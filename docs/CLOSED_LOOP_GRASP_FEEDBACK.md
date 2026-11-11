# Bounded measured slip and object-tilt feedback

`arm_replay_pilot` accepts `closed_loop_grasp_feedback` containing a pinned
`baseline: {path, sha256}` and explicit `controller` settings. It consumes a
complete continuous live-finger source and preserves its reference, geometry,
pressure targets, motor/torque limits, clocks and physical acceptance gates.
Only output, purpose, this feedback and the explicit stationary diagnostic mode
may change. Source scene/state hashes are checked. Existing wrist controllers,
static correction probes and pressure actors cannot share this arm owner.

`SlipTiltFeedback` is the pure controller; `ClosedLoopGrasp` binds it to CPU
MuJoCo replay without a learner or mandatory GPU dependency. Settings are:

```json
{
  "slip_gain_s_inv": 1.5,
  "tilt_gain": 0.5,
  "height_gain": 0.0,
  "filter_gain_s_inv": 3.0,
  "max_offset_m": 0.006,
  "max_angle_rad": 0.06,
  "max_speed_m_s": 0.012,
  "max_angular_speed_rad_s": 0.12,
  "minimum_contact_n": 1.0,
  "maximum_reference_error_m": 0.02,
  "maximum_reference_angle_rad": 0.17453292519943295,
  "observation_period_s": 0.02
}
```

The source's first recorded sample at/after acquisition completion defines the
fixed object position in each actual wrist frame. Subsequent drift from those
anchors supplies a bounded integral wrist translation. Anchors never reset on
contact loss, hold entry or slip. Downward slip integration is blocked when the
object is already below its reference. This guard does not forbid coordinated
wrist motions needed to apply a restoring moment.

Object orientation error supplies an opposing common rotation. Both nominal
wrists move about the measured object-body origin, with matching orientation
corrections, rather than rotating independently about each wrist. Coordinated
tilt updates require contact on both hands. The optional height term requests
only upward translation when the object is below its reference; original return
commands still control descent. It changes no object state or reference.

Each wrist offset is filtered and limited to 6 mm / 0.06 rad with explicit speed
bounds. New corrections freeze on missing contact or excessive reference error.
Contact is measured total hand/object normal force; it does not certify an
opposed grasp or force closure. Independent task diagnostics retain those checks.

The observer copies live physical state/commands/warmstart into separate data and
solves coherent poses/contacts there. Measurements are held between observation
ticks; the reference and command clock continue at the native feedback rate.
Consequently observations can be up to one observation period old. This is
privileged simulation sensing, without a hardware estimator/noise model.

Wrist twists are projected through damped joint IK, preserving native joint
margin and arm-speed bounds. Requested commands and their preceding-command
midpoints are screened against robot/environment collisions in scratch data.
Intended object contact is excluded; arm and robot penetration thresholds remain
the source thresholds. A guard failure aborts and rejects the replay instead of
issuing the unsafe command. These are sampled checks, not a continuous certificate.
The ordinary physical penetration/torque/velocity checks also remain active.

At capture end, continuous feedback stops acquiring corrections and smoothly
recovers the accumulated offsets to zero by scheduled release, even on contact
loss. Configuration validation requires enough return time for that recovery to
respect the declared speed bounds. The source table-support gate, opening and
withdrawal remain the release owner. Stationary diagnostics retain their offsets
for the complete hold with release disabled.

Run all three feedback gains at zero first. This retains observation and anchor
logic while bypassing arm correction arithmetic, allowing exact archived-state
regression. Compare every state/control/time sample, scene, diagnostic record,
return/release metric and rejection reason. A successful zero test does not make
the underlying physical task successful.

`closed-loop-trace.json` records fixed anchors, offsets, slip/tilt errors, contact
forces and update/freeze counters at saved-state times. Result metadata adds
travel peaks, collision-guard outcomes, recovery state and source snapshots.
All physical verdicts remain independent of this telemetry. Results declare
`promotion_eligible=false` and `training_run=false`; a later learned actor needs
its own action integration, zero-path regression and policy-aware promotion.
