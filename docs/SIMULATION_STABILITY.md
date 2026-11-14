# Diagnosing contact instability and transport screens

When explaining a rejected transport plan, identify the enforced decision,
not simply the largest overlap printed in its diagnostics.
`agentic_policy.transport_diagnostics.rejection_evidence` selects the governing
contact-aware or loaded-preview rejection when enabled; otherwise it reports
the legacy stationary/carried failures. Loaded soft-pad overlap may be logged
by the stationary screen while an unrelated rigid finger-link overlap actually
blocks admission. The helper preserves the witness and geometric assumptions,
does not execute motion, and never presents a forecast as observed live contact.

Preserve the original controller, scene, evaluator and failed result before
changing numerical settings. A reset is a failed simulation, not a successful
release or return home. An unchanged finer-timestep run is a sensitivity test;
it is not automatically a better or accepted controller.

## Every-step capture without perturbing live dynamics

`controller_actor_training.mujoco_step_trace.MujocoStepTrace` wraps individual
`mj_step` calls for one model/data pair in a single-threaded diagnostic process:

```python
with MujocoStepTrace(model, data, output, window_s=1.0):
    run_controller()
```

Use an output directory that does not exist. The bounded trailing window records
every integration step, actual actuator commands, forces, cached contacts,
constraint fields and the complete pre-step `mjSTATE_INTEGRATION`, including
warmstart. A callback can capture controller memory such as pressure offsets.
The wrapper reads cached fields; it adds no forward calls on live data. Cached
post-step dynamics need not correspond to the post-integration configuration.

On a clock discontinuity, new warning or nonfinite state, the trace freezes and
recomputes the failing input in a separate `MjData`. Inspect that scratch result
before changing gains or attributing the failure to grip strength. Saved
integration state is sufficient for the simulator input, but continuing a
controller also requires its controller memory and program position.

`audit_contact_geometry` checks contact finiteness, frame orthogonality and gross
contact-point displacement outside finite geom bounding spheres. It accounts
for separation/penetration and contact margin. Planes and unsupported types
receive finite/frame checks only. Call it on position-consistent scratch data;
it never deletes, repairs or admits contacts. This is a diagnostic sanity check,
not validation of the reconstructed surface or a proof of force feasibility.

If collision generation already emits implausible points before solving, a
solver failure can be downstream of that geometric defect. Compare collision
algorithms only as explicitly labeled counterfactuals in a separate model.
Finite forces at one saved state do not establish a successful full rollout.
Do not disable contacts or loosen acceptance thresholds to hide the defect.

## Matched numerical sensitivity

Freeze feedback period, physical targets, pressure settings, scene, initial
state and acceptance criteria. Change only the integration timestep in a
separate input bundle; choose timesteps that divide the feedback period exactly.
Regenerate only timestep-dependent clock representations, recording any
floating-point differences. Verify compiled physical parameters are unchanged.

First reproduce the original saved states and actuator commands exactly with
tracing enabled. Then compare complete nominal results at both timesteps for
both the candidate and retained baseline. Distinguish numerical reset from a
finite trajectory that loses its grasp or hits an obstacle. A baseline passing
at only one timestep remains a useful comparison, with unresolved numerical
sensitivity. Two timesteps alone do not demonstrate convergence.

## Transport collision assumptions

`agentic_policy.transport_screen_audit.audit_transport_screen` samples the same
joint path and compares a stationary object with a hypothetical rigidly carried
object, using scratch position calculations. It separates robot/object,
object/environment, robot/environment and robot/self collisions. The free
object must be attached to the world. All durations align to feedback ticks.
The helper neither advances physics nor admits the path.

Moving the hand while leaving a grasped object stationary can create contact
overlap that would not occur with retained grasp. Conversely, assuming rigid
carrying can hide the consequences of slip. A screen disagreement therefore
requires review of the witness pair, loaded retention and external clearance;
it is not permission to bypass collision checks. A subsequent contact-aware
executor must preserve obstacle/self checks and live physical limits, and must
be validated as a new interface condition.

## Opt-in contact-aware execution

The default `transport_screen_mode: stationary_and_carried` preserves the
original adapter. `contact_aware` uses configured pads with observed contact
force and an acquisition reference to distinguish intended pad/object pairs.
Only those pairs are exempt from the stationary-object hypothesis. Carried
object contacts, proximal links, palm, robot self and environment collisions
remain screened at the original penetration limit. Joint, velocity, clock and
horizon validation still applies. Plans are retained by the trusted host and
rechecked before execution; the lower-level screened executor is not an RPC tool.

The optional `contact_aware_preview` mode can resolve a carried **intended-pad-only**
kinematic rejection with a short loaded simulation. Each forecast is at most
0.5 s, with at most 64 forecasts per session including execution rechecks. The
forecast copies complete MuJoCo data and controller memory, including pressure
offsets. Model parameters are shared read-only. No reset, weld, teleport, command
change, extra live step or pause is added. The existing wall deadline still
applies. External, self and non-pad collision rejections cannot use this route.

Forecasts run the same actuator loop with hard physical penetration, nonfinite
and contact-loss checks at the existing feedback observations and endpoint.
Strict hold-relative drift is recorded as quality evidence, rather than silently
becoming a new nominal loading abort rule. Candidate phase-specific guards and
the independent evaluator remain in force. Reports preserve failed forecasts
and count this planning computation separately from live trial steps. An
executed preview records exact equality of predicted versus actual final qpos,
qvel, actuator commands and time. This is same-model prediction, not independent
validation, a robustness guarantee or permission for hardware control.

The explicit `verify_grasp` hold uses the same opt-in loaded screening path as
transport. Its duration, nominal joints and original hold-quality checks stay
unchanged. Ordinary acquisition holds retain their original behavior. A hold
forecast is counted in the same budget and records forecast/live equality;
kinematic nominal-posture overlap alone must not bypass or fabricate a loaded
hold result.

## Review between proposals

Use `mode: review` for a campaign where each proposed change needs human input.
`advance` prepares a candidate and stops before physics; `submit` runs that one
candidate and returns without requesting the next author revision. At that
point inspect its video, actual joint motion, contacting link names, forces,
slip and collision witnesses before the next `advance`. Automatic mode has no
such mandatory per-proposal checkpoint. See [author campaigns](AUTHOR_CAMPAIGNS.md).

A three-pad thumb/index/middle grasp is a legitimate candidate. Fingers held
extended are not disabled; their proximal links and palm still need clearance.
Review the opposition, moment resistance and inversion clearance rather than
rejecting a grasp for appearance alone. A short lift is evidence for initial
retention, not proof of inverted carrying. Repeated numerical failures should
trigger a bounded simulator/interface investigation before further author calls.

MuJoCo documents [integration state and replay](https://mujoco.readthedocs.io/en/3.8.0/programming/simulation.html)
and [contact generation](https://mujoco.readthedocs.io/en/3.8.0/computation.html#collision-detection).
Pin the simulator version; collision defaults can change between versions.

### Loaded obstacle witnesses and uncertain geometry

`controller_actor_training.obstacle_clearance_audit.clearance_witness` reports
signed-distance witnesses between explicit geometry sets at saved configurations.
It never steps physics or changes the model, exclusions or acceptance limits.
A sampled clearance check is not a continuous swept-path certificate. Keep the
original per-step contact forces for load diagnosis; a position-only replay does
not reproduce loaded forces.

Validate narrow-phase distance results before interpreting them. A zero distance
with separated witness endpoints, endpoints outside the geometry bounds, or a
nonfinite result is **unknown**, not proof of clearance or contact. The helper
reports these invalid queries separately and withholds a complete-clearance claim.
This catches observed mesh-query failures even when the saved rollout remains
finite. It does not silently switch collision backends.

Map contact colliders back to the observations that created them.
`retargeting.observed_colliders.box_observation_evidence` records occupied samples
and nearest distance per view in a shared metric frame. Keep fit and validation
views separate. Missing valid depth is not free space; thin-feature boundaries
can mix foreground and background disparities. A modeled contact can be real in
simulation while its physical boundary/thickness remains uncertain. Retain the
obstacle unless source evidence supports a documented correction. Do not remove
it to satisfy an off-support hold gate.

### Whole-object clearance targets

`retargeting.clearance_envelope.vertical_clearance_target` bounds the lowest
vertex of the complete carried collision geometry under a declared tilt reserve.
It uses the geometric displacement bound `2*r*sin(theta/2)` and adds explicit
tracking and separation reserves to an obstacle-height envelope. This avoids
assuming that a high object center implies a clear cup base or rim. The result
is an endpoint planning target, not path admission: insertion, full hand/arm
sweeps, live load response and the original hold criteria must still pass.
Margins are design assumptions, not measured uncertainty confidence intervals.

### Rack support is not all-environment support

A support signal computed over a configured placement surface may omit the table
or another obstacle. Do not use a zero value of that restricted signal as proof
that a carried object is airborne. Check actual object clearance and the relevant
fixed-surface reactions before a lateral retreat or rotation; commanded wrist
height alone can mask object rolling/slip.

`controller_actor_training.traced_contact_forces.grouped_object_forces` audits
original saved solved-contact wrenches without re-solving dynamics. It handles
the geom1/geom2 force sign and contact-to-world rotation, preserves unmapped
counterparts as `unclassified`, and rejects malformed contact frames. Keep the
cached-contact timestamp convention explicit. This diagnostic does not modify
existing support definitions or acceptance limits, and does not certify grasp
stability from pad force magnitudes alone.

## Long loaded-hold diagnostics

A hold request can fail admission because it exceeds the loaded preview's
segment horizon, even before any physical contact loss. Distinguish that result
from an executed hold that loses contact. For a longer diagnostic, use consecutive
admitted segments within the preview horizon, preserving the same live state,
pressure memory and acquisition anchor. Aggregate the entire requested interval;
a good final segment does not erase earlier drift or interrupted contact. Keep
these diagnostic pauses separate from the nominal task's timing. Report the
original strict hold verdict separately from short-window movement and contact
count. Verify trajectory parity where comparing different segment partitions.

## Sliding versus patch migration

`controller_actor_training.traced_contact_motion.audit_trace` restores saved
integration inputs into separate simulation data and reuses `ContactPatchMotion`
to measure tangential material velocity, rolling-compatible motion and contact
centroid migration separately. It never steps physics. Supply the original
configured model, including base placement and collision options. Check its
reported normal-force reproduction error before trusting the contact audit.
Saved post-step contact fields correspond to pre-integration kinematics, not the
post-integration joint positions. This audit uses the matching integration input.

The result covers only the saved trace window. A bounded tail trace cannot
establish whole-trial retention. Its sample-gap rule avoids integrating across
missing observations. Strided trace audits retain the final available input
sample so downsampling does not omit the end of the recorded window. A later
terminating observation can still differ from the last pre-integration contact;
report both with their respective times. Nonzero pad force alone does not establish no-slip contact;
contact centroid migration alone is also not a material-slip measurement.
Neither this diagnostic nor a completed hold replaces the original acceptance
criteria or proves retention during inversion.
