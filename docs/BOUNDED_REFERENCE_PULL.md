# Experimental bounded body-reference correction

`SonicJointResidualActionConfig.reference_pull_gain` enables an optional fixed
correction toward reference joint targets. The default is zero and preserves
the existing SONIC/learned-residual path. This is an experiment for investigating
joint tracking, not a trained grasping or navigation policy.

After SONIC and any learned residual form a body-joint target, add
`clip(gain * (reference_target - current_target), -bound, bound)` to that target.
`reference_pull_max_rad` defaults to 0.15 rad; accepted configuration bounds are
gain 0–1 and correction limit >0–0.3 rad. Explicit coordinate/DOF maps preserve
nontrivial joint order. The effective target and action-target history are updated
together. Finger targets and fixed hand timing remain independently controlled.
No root pose, external force or object assistance is added.

Compare against zero gain under the same reference, model assets, inertial model,
control clock and evaluator. Measure the actual palm's distance to its intended
object-relative grasp point using the *current* physical object pose. Also record
upright hand contact, pelvis/balance and task metrics. A smaller joint error is
insufficient evidence of successful reach or grasp. Reject improvements obtained
only after a fall. Repeat any promising candidate before retaining it.

This option can compete with the prior's balance behaviour and is disabled by
default. It does not provide absolute root-position feedback or certify that a
reference gait is physically trackable. Do not enable it across new tasks merely
because one local diagnostic improves.

Use `controller_actor_training.reach_audit.summarize` on saved physical and
kinematic `[frame,body,xyz+xyzw]` arrays with explicit body IDs and boolean contact
phases. It computes each intended grasp point in reference-object coordinates,
then carries it by the actual object pose before measuring palm error. Optional
frame masks can exclude fallen/invalid states. The diagnostic is invariant to a
common rigid change of world coordinates; it neither changes states nor labels
proximity as grasp success. Retain stability and object-tracking gates even when
this reach error improves, and require all verification repeats to pass.

Use `reach_audit.summarize_contacts` with saved contact counts, their explicit
hand-slot mapping, intended contact phases and an upright/valid-frame mask to
distinguish incidental touches from task-phase contact. Increasing total contacts
alone can reward an early collision that never helps manipulation. Report both;
do not reinterpret a screening contact increase as grasp acquisition or silently
change predeclared experiment gates after observing results.
