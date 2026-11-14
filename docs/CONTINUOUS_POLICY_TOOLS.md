# Continuous policy tools for open policy comparisons

`controller_actor_training.continuous_policy.ContinuousPolicy` exposes a broader
policy interface over one fixed MuJoCo scene, separate from the zero-edit
`seating_graph_replay` adapter. Freeze the scene, initial state, comparison
controller, deadlines and outcome criteria before authoring. The author may use
project source, planning libraries and GaP skills to choose grasps, postures,
paths, timing and recovery logic. Scenario parameters and outputs stay downstream.

| Tool | Operation |
| --- | --- |
| `robot.describe` | Named joints, limits, initial home, pad frames and task |
| `robot.observe` | Actual/commanded joints, object/wrist poses, pad positions/normals and forces |
| `robot.plan_wrist` | Scratch-state arm IK; does not advance physics or admit a path |
| `robot.fit_hand` | Scratch-state fitting of declared pads with native joint bounds |
| `robot.screen_path` | Sampled collision and command-rate checks with current object pose |
| `robot.execute_joint_path` | Joint targets executed through the existing actuators |
| `robot.hold` | Continue the current command through physics |
| `robot.set_pressure` | Optional bounded correction on declared finger pads |
| `robot.release` | Screen opening after sustained supported placement |

All configured arm and finger actuators are commandable. Automatic pad fitting
and pressure control cover only explicitly calibrated frames. Current pressure
targets must all be positive and at most 5 N, or all zero to disable correction.
Targets are not guaranteed measured force ceilings.

Paths specify every robot joint, finite native-range angles in radians and
positive durations aligned to feedback ticks. Each segment uses cubic
rest-to-rest interpolation: a waypoint intentionally has zero commanded
velocity. Segment timing belongs to the proposed policy. Graph tool handoffs
add no simulated dwell, reset or extra clock steps. Planning/observing does not
advance physics. This is offline synchronous simulation, not real-time hardware.

Physics timestep, actuator gains/force limits, friction and geometry stay fixed.
Position actuators receive bias compensation and arm velocity feedforward.
There is no object-pose setter, reset or scene-edit tool. The host owns simulation,
deadlines and evaluation. `policy_tool_process.PolicyToolProcess` and
`RemotePolicyTools` provide typed JSON calls to a separate persistent process;
the GaP environment need not contain MuJoCo. `register_policy_tools` supports
direct and remote hosts. The host must separately enforce filesystem/process
isolation; RPC alone does not sandbox arbitrary Python.

## Broad code access

Give the author readable source, examples and schemas, including real2sim2real,
GaP, ASL and installed skill libraries. Pin revisions and local modifications
by content hash. Source availability is distinct from runtime readiness: weights,
dependencies, robot assumptions and backends still apply. Compatible geometry
tools can run alongside `robot.*`; other backend motion functions need translation
into this fixed interface. Do not substitute a demo robot or scene.

For a from-scratch comparison, algorithms and common reconstructed geometry are
available, while successful manual grasp/path parameters and reference evaluation
data stay evaluator-side. Baseline-seeded comparisons are separately labeled.

## Evaluation and limits

`placement_outcome` independently checks stable supported, inverted, released
placement and same-home return from physics records using downstream thresholds.
It does not trust a graph exit status or author success flag. Outcome completion
is separate from strict quality, robustness, reference scores and control acceptance.

Path screening samples at the feedback interval with the current object pose;
it is not a continuous collision certificate or loaded-contact prediction.
Execution monitors solver health every physics step and penetration at the
configured sample interval. A configurable minimum object COM height defaults
to -0.1 m in the upright metric world, retaining the diagnostic fall stop.
Downstream evaluation must retain uncertainty-envelope,
contact-quality and reference-scoring audits. Conditional geometry is not verified
reconstruction.

Tests cover invalid-command atomicity, persistent-clock partition parity,
nonzero dynamics, final-state capture between sample ticks and rejected incomplete,
unsupported or unreleased outcomes. A downstream full-horizon GaP negative control
must report task failure even if its graph succeeds without manipulating the object.
