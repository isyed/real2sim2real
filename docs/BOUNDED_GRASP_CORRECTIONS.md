# Bounded wrist and posture authority probes

`arm_replay_pilot` accepts an optional `bounded_grasp_correction` for a pinned
continuous live-finger baseline. This provides a small, declared robot-command
experiment before assigning more actions to a learner. Every probe retains the
source physical/controller settings, task limits and raw object reference.

The specification contains:

```json
{
  "baseline": {"path": "/data/frozen/result.json", "sha256": "..."},
  "vectors_object_m": {"right": [0, 0, 0], "left": [0, 0, 0]},
  "rotation_vectors_object_rad": {"right": [0, 0, 0], "left": [0, 0, 0]},
  "finger_joint_offsets_rad": {}
}
```

Wrist translations are bounded by 6 mm per wrist and rotations by 0.06 rad.
Vectors use the reconstructed reference object's axes at each timestamp;
rotations pivot about each wrist origin. They do not directly move the object.
Named finger-flexion changes are bounded by 0.06 rad per joint, 0.12 rad per
hand and 0.2 rad/s correction speed. Physical joint and actuator limits remain
unchanged. The nominal finger correction enters the existing live pressure
servo; a second controller never writes those actuators.

The correction rises smoothly during the existing closure interval. During
continuous return it recovers to zero between capture end and scheduled release.
The wrist planner requires equality with the original commands from the last
waypoint before release onward, preventing interpolation from extending the
correction into release. Original release posture, table-support gating and
withdrawal remain owned by the existing controller. A stationary diagnostic
instead retains its corrected suspended targets for the entire original hold;
its release is disabled by the existing motion-condition mechanism.

The runner pins the baseline result/scene/states and rejects controller/physics
retuning or corrected-source stacking. Only output, purpose, the explicit probe
and demonstrated-versus-stationary diagnostic mode may change. A pressure actor
cannot be stacked with this first probe implementation.

Wrist IK runs in scratch data, with native joint margin/speed admission and a
0.15 rad bound on the combined arm-joint change. Position/orientation residuals
must be <=0.3 mm/0.003 rad. Both hands then pass the existing complete sampled
arm/environment collision screen with the new nominal finger commands. The
planner writes no physical state. These are sampled checks, not continuous
collision certification; the complete free-object rollout remains necessary.

Outputs include `bounded-correction-path.json`, `bounded-command-targets.npz`,
the ordinary state/result artifacts, and correction metadata with
`promotion_eligible=false`. Physical acceptance still comes from the unchanged
task evaluator. A successful relative improvement does not supply a stationary
promotion certificate, payload/friction robustness or hardware validation.

Always run an all-zero probe first and compare every saved qpos/qvel/ctrl/time
sample, scene, diagnostic record, return/release metric and task verdict against
the pinned baseline. Compare moving probes and paired complete stationary holds
separately. Source snapshots distinguish early probes from later added guards.

Before residual RL, measure whether bounded corrections materially affect the
failure and preserve other gates. Static probes measure only the declared
directions/postures; failed static probes cannot rule out causal, time-varying
feedback or all possible grasps. A later actor needs its own action-space
integration, collision checks and exact zero-action verification.
