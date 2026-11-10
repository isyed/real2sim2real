# Object-frame wrist coordination

`controller_actor_training.wrist_coordination` adds an optional, bounded common
motion to two force-limited wrist fixtures. It builds on the
[articulated support workflow](BIMANUAL_SUPPORT.md). The original contact plan,
object, table, actuator limits and uncoordinated baseline remain available.
This is a simulator controller experiment, not an arm planner or trained policy.

## Controller contract

Declare `object_frame_coordination` alongside an enabled `support_controller`:

```json
{
  "translation_gain_s_inv": 1.0,
  "rotation_gain_s_inv": 2.0,
  "max_translation_speed_m_s": 0.02,
  "max_rotation_speed_rad_s": 0.2,
  "max_translation_m": 0.025,
  "max_rotation_rad": 0.2,
  "joint_limit_policy": "project_common"
}
```

These are example experiment settings, not gains validated for another object.
The object origin at hold entry defines a fixed pivot and horizontal position
target. The original gravity-settled object orientation is the rotation target.
Feedback integrates bounded horizontal and angular corrections from the current
object pose. It does not move the goal to follow a slipping or falling object.
Vertical clearance and support-load feedback remain separate.

For each uncorrected wrist frame `(R_i, p_i)`, the same correction `(R_c, t_c)`
around pivot `o` gives:

```text
p_i' = o + R_c (p_i - o) + t_c
R_i' = R_c R_i
```

This rotates the wrist positions around the object as well as their orientations,
preserving the relative transform of the two input wrist commands. It does not
guarantee that the force-limited physical hands achieve those commands, retain
contact, or preserve their relative transform as the separate load loop changes.

Approach, closure, lift and support insertion retain the previous controller.
At hold entry, the primary hand's independent orientation correction is frozen;
subsequent orientation updates belong to the common controller. The common
transform is applied before support-hand actuator bias compensation. Its
kinematic residual reports the geometric target, not physical tracking error or
the later bias-offset setpoint. Actual saved states and contacts are evaluated
separately.

The adapter requires world-parent bodies with six limited, zero-reference joints:
world-aligned XYZ slides followed by intrinsic XYZ hinges. It checks that
contract and maps world commands back into each fixture. Scratch kinematics do
not write physical joint positions, object state, external forces or object
actuators. Other fixture layouts and whole arms require an explicit IK adapter.

## Joint limits and failure handling

The default `reject` policy records an infeasible common command as a failed
trial. Actuator commands are bounded, and the resulting geometric discrepancy
remains visible. Independently clipping the two wrists cannot count as a valid
common motion.

`project_common` instead backtracks the common update to a pose both wrists can
reach. It first tries to preserve feasible horizontal correction while limiting
rotation, then bisects the remaining update. The integrator keeps only the
accepted correction. If the previous correction has itself become infeasible as
the independent input commands change, the trial is rejected. Per-trial records
include projected/update counts and maximum realized command pose errors.
This is a conservative feasibility projection, not a constrained contact or
wrench optimizer; one limiting joint can block other useful angular motion.

## Evaluation and reuse

`grasp_robustness` accepts named `object_frame_coordination` variants in a pinned
comparison. An omitted/null setting preserves the baseline. Shared rotation
alone can be tested with zero translation gain. Frozen validation rejects changed
coordination gains or limit policy. A diagnostic extension of a failed controller
cannot promote it, even if some longer trials pass.

Existing clearance, opposing-finger contact, wrist-relative drift, rotation,
support-load and penetration gates remain active. Coordinated trials additionally
require at most 20 mm horizontal object drift in the fixed world and command pose
errors below 1 micrometre / 1 microradian. World drift is also displayed for saved
baseline trials without changing their historical verdicts. This prevents two
wrists moving together with a drifting object from hiding task displacement.

Compare contact retention and both wrist-relative drifts alongside world motion.
Lower object drift can coexist with loss of opposing contact, larger in-hand
motion or joint-limit saturation. Such a controller must not replace its baseline
solely because its video or position error looks better.

Object pose and contact forces are privileged simulator observations in this
pilot. Whole-arm reachability/collisions, robust contact planning, deployable
state estimation and task-specific policy training remain separate work.
`tests/test_wrist_coordination.py` covers rigid relative-frame preservation,
feedback direction/bounds, independent world-motion auditing, frozen settings,
and optional MuJoCo fixture mapping/limit projection without object assistance.
