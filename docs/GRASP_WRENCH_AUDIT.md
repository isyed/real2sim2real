# Instantaneous grasp force and torque audit

`controller_actor_training.grasp_wrench_audit` investigates whether the current
contact geometry can transmit a declared six-dimensional object load within the
simulator's friction and actuator envelopes. It reads pinned complete continuous
physics archives; it does not advance control, modify physical states, replace
the reference, promote a controller or train an actor.

Run this diagnostic before assuming a slipping grasp needs greater pressure.
Separate airborne support from table-supported load transfer. After touchdown,
hands can push downward while the table pushes upward; contact drift in that
interval is not automatically an inability to lift the payload.

## Inputs and frames

The CLI takes `--config` with a pinned `result: {path, sha256}`, fresh `output`,
`object_body`, named `hand_roots`, `start_s`, `end_s`, `sample_period_s`,
`critical_times_s`, `witness_time_s`, odd `derivative_window_frames` and
`sensitivity_window_frames`, `lp_time_limit_s` and finite `deadline_s` (at most
900 seconds). The adapter requires a separate free object, elliptic contact
cones and one positive-gain, unit-gear, fixed-gain affine hinge actuator per
audited joint. Unsupported embodiments fail rather than silently omit motors.

The scene, qpos/qvel/ctrl archive, arm-path audit, source geometry/configuration,
command ancestry and reconstructed reference are hash checked. Source filenames
and dataset-specific paths belong in a downstream configuration. Optional MuJoCo
imports occur only inside the runtime stage.

Every load is `[Fx, Fy, Fz, Mx, My, Mz]` in world coordinates, in N and Nm about
the **actual object center of mass**, using body inertial position and principal
inertia orientation. Contact forces are signed onto the object. Their opposite
reaction on the hand maps through each contact point's translational/rotational
Jacobian to finger **and arm** actuator torques. Tangential, torsional and rolling
friction components are retained when present in the model's contact dimension.

Each archive sample is copied to scratch simulator data and re-solved with zero
warmstart. Action/reaction reconstruction must close the object wrench balance
and robot torque balance. These are coherent saved-state force solves, not
additional trajectories or independent physical measurements. The implementation
follows MuJoCo's [contact frame and friction model](https://mujoco.readthedocs.io/en/stable/computation/index.html)
and [contact-force/Jacobian API](https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html).

## Two load questions and four actuator envelopes

Gravity asks whether the free object could be held at the observed contact poses
with quasistatic robot gravity/passive forces and no assistance from other robot
constraints. It is an instantaneous screen, not a completed stationary hold.

Reference motion uses `m(a-g)` and `I alpha + omega × I omega`. Local cubic fits
estimate world derivatives from the unchanged native reconstructed poses and
timestamps; a second window reports derivative sensitivity. Robot inertia and
other measured robot constraint reactions are frozen at the observed dynamic
state. This is conditional capacity, not a new self-consistent motion solution.

| Envelope | Normal cap per anatomical contact group | Available commands |
|---|---:|---|
| `motor_25N` | 25 N | Native motor force and control ranges |
| `goal_10N` | 10 N | Native motor force and control ranges |
| `feedback_inner` | 10 N | Inscribed joint-offset box inside the existing per-pad correction ball |
| `feedback_outer` | 10 N | Circumscribed joint-offset box around that correction ball |

The 10/25 N caps are **analytic sensitivity envelopes, not manufacturer limits**
or guaranteed contact-force commands. Multiple contacts on one finger share its
cap. Native force/joint-actuator limits remain enforced. Position-control bounds
map through the existing PD gain, position and velocity bias, including force
saturation. An `n`-joint pad uses inner half-width `budget/sqrt(n)` and outer
half-width `budget`. The source controller's existing offset defines the nominal
command; fitting a force witness does not establish that its feedback law can
produce that witness at the required rate.

`wrench_feasibility` constructs inner contact rays and outer supporting facets of
the elliptic friction cone, then solves bounded LPs with
[SciPy HiGHS](https://docs.scipy.org/doc/scipy/reference/optimize.linprog-highs.html).
An inner-cone feasible witness is sufficient for this ideal static contact model.
Outer-cone infeasibility rules out that declared envelope. Intermediate results
remain unresolved. An outer feedback box or outer friction cone alone cannot
establish feasibility inside the actual correction ball and exact cone; witnesses
are checked against both. Solver timeouts remain unresolved, not infeasible.

The minimum-normal-force LP directly checks the requested load. The capacity LP
maximizes a multiplier on its **entire force/moment vector**, with robot background
torques unscaled. A maximum multiplier above one does not by itself imply the
nominal load is feasible: lower torque bounds can also require a minimum contact
load. Always inspect the direct feasibility result and unsolved cases.

External object forces are reported separately and never granted to the hand-only
LP. After touchdown that LP is explicitly counterfactual. Do not interpret it as
the actual force distribution during placement.

## Motion and review

`grasp_wrench_review --config ...` compares pinned audits without rerunning
physics. Its config contains `cases: [{label, audit: {path, sha256}}]`, a fresh
`output`, `motion_windows: [{label, start_s, end_s}]` and `trend_window_s`.
It writes `summary.json` and a self-contained `review.html` with airborne versus
external-support counts, sampled touchdown, signed vertical forces and wrist
position variation.

`sampled_motion_variation.position_variation` removes a local cubic position trend
from uniformly sampled metric positions and excludes filter edges. Its RMS/peak
residual describes real saved motion, including deliberate changes in the path.
It is **not** tracking error or proof of unwanted oscillation. A 50 Hz archive
cannot identify motion above 25 Hz. Low-frame-rate playback, object-following
cameras and collision-mesh rendering can exaggerate visible jerkiness. The review
renderer replays saved qpos; rendering does not remove physical control motion.

Next isolate planned/commanded versus measured motion and inspect contact/pressure
traces before attributing sway to a specific feedback loop. A real robot may have
different compliance, sensing and delays; simulator motion cannot establish that
oscillation will disappear on hardware.

## Limits and decisions

Force witnesses do not enforce sliding/contact-mode compatibility, normal
compliance, contact creation, sustained retention, rate limits, full insertion,
return or release. Uncertain geometry, friction and hardware adapter assumptions
remain uncertain. Large force capacity with measured drift should redirect the
investigation toward seating and control/load transfer rather than certify the
grasp. Preserve all original verdicts. A new candidate still requires acquisition,
appropriate complete holds, loaded motion and the unchanged acceptance limits in
[control readiness](CONTROL_READINESS.md).
