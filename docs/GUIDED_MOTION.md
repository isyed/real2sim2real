# Reconstructed motion guidance and demonstration strategy

`controller_actor_training.guided_motion` transfers reconstructed object-motion
increments through two robot wrists after an existing grasp is established.
`egocentric.demonstration_strategy` reviews synchronized approach/lift images and
estimated object/table motion before proposing a new contact strategy. Neither
module establishes accurate human finger reconstruction or full-task imitation.

## Bounded motion transfer

`MotionReference` accepts contiguous frame IDs, increasing capture timestamps
with gaps no greater than 100 ms, and metric rigid object poses. Translation is
linearly interpolated and orientation uses quaternion interpolation. The source
timing, scale and world axes remain fixed. Values outside the interval retain
the first/last pose; missing intervals are rejected, not interpolated across.

The runner pins a successful grasp, its saved states, prepared scene and source
motion. It repeats the original controller and tests one guided candidate with
the same scene, material assumptions, actuator limits and physical gates. An
optional archived baseline-state hash checks exact deterministic regression.
The two variants have the same complete duration. The reference must leave at
least the task's required final hold duration after motion ends.

For source object position/rotation `(p(t), R(t))`, guidance uses
`dp(t) = p(t) - p(0)` and `dR(t) = R(t) R(0)^T` in the upright capture world.
The achieved object pose at support entry supplies a fixed simulation anchor.
Both wrist commands undergo the same rigid motion about that anchor. This
preserves their commanded relative frame before the existing support-load loop
and actuator bias compensation; it does not guarantee actual grasp retention.
Baseline clearance/orientation feedback sees the object with this commanded
motion removed, so it responds to tracking error rather than fighting the
demonstrated motion. Gains and physical limits remain unchanged.

This is explicitly `world_increment_after_grasp`. It does not align or correct
a floating reconstructed initial pose, reproduce the human approach, or replay
the original absolute object trajectory. Capture-relative rotation is applied
to the achieved grasp orientation without rotating the world's gravity axis.
Original capture reconstruction and initialization discrepancies remain separate.
The transfer uses only force-limited robot actuators. Object poses, gravity,
welds and external object forces are never used to enforce motion.

Only the checked Cartesian XYZ-slide/XYZ-hinge wrist fixtures are supported.
A fixture limit aborts the candidate rather than independently clipping the two
hands and claiming successful common motion. The preexisting common wrist
coordinator cannot simultaneously own these commands. Whole-arm IK and collision
planning require a separate adapter.

`guided_motion_pilot` reports physical success and reference fidelity separately.
A stationary object can pass lift/hold checks and fail motion fidelity; a moving
object can follow its target closely while slipping in the grasp. Neither result
can override the other. Motion error is computed against the transferred target,
not challenge ground truth, and is not an official AUC/MPPE score. Exact simulator
feedback is privileged. Human imitation, robust control and hardware execution
remain unvalidated even if this nominal diagnostic passes.

```bash
python -m real2sim2real.controller_actor_training.guided_motion_pilot --config pilot.json
python -m real2sim2real.controller_actor_training.guided_motion_review \
  --evaluation run/evaluation.json --output run/review
```

The pilot config supplies `scene_config`, `parent_trial`, `object_reference`,
`first_frame`, `last_frame`, `hold_s`, a fresh `output`, `deadline_s`, source
hashes, explicit translation/rotation/speed bounds and separate fidelity limits.
Optional `hand_hypotheses` and `detector_landmarks` audit wrist evidence; they
do not enable human wrist control. The present adapter implements object guidance
only. Point trackers may omit wrists while a detector has partial wrist labels;
neither a zero track weight nor an admitted fitted hand proves absence/presence
of independently accurate metric wrist evidence. Human approach motion must not
be applied indiscriminately after the robot has already grasped the object.

## Observation and strategy audit

`demonstration_strategy` loads hash-bound synchronized capture streams, accepted
reconstruction and its accepted upright support. It produces calibrated
object-centered crops in up to three views, retaining source pixel boxes, frame
IDs and timestamps. It reports reconstructed clearance, translation and rotation
without resetting the starting height. Optional object-local probes expose how
the same surface points differ between reconstruction and a separate settled
task. Probe heights are estimates, not observed contact locations.

```bash
python -m real2sim2real.egocentric.demonstration_strategy --config audit.json
```

The configuration supplies reconstruction/capture/support paths and hashes,
`frame_count`, up to 24 `review_frame_ids`, a metric `crop_radius_m`, a fresh
`output` and `deadline_s`. Keep scene-specific visual interpretations downstream,
labeling assistant inspection separately from verified anatomical labels. A
visible two-hand arrangement can inform candidate contact regions without
revealing hidden contact normals, forces or all finger joints. A plausible
strategy does not validate its reconstructed collision surfaces.

## External adapters

Do as I Do's Mink IK remains a potential backend for independently supported
wrist/fingertip targets or later arm integration. These Cartesian fixtures have
an explicit mapping, so adding another IK solver does not resolve missing wrist
evidence or contact slip. Its sampling MPC is a candidate for constrained motion
or contact refinement, after defining phase-aligned targets and independent
unassisted replay checks. It has not been integrated or executed by these modules;
no runtime imports from neighboring projects are introduced.

Tests cover capture timing, missing intervals, fixed anchors, feedback-frame
conversion, stationary-object fidelity failure, fixture-limit rejection and
preservation of physical/object states. The strategy test keeps nonzero starting
clearance and never infers contact from height alone.
