# Arm feasibility before physical integration

`controller_actor_training.arm_path_audit` checks whether saved wrist-fixture
trajectories can be followed by an explicit arm embodiment. It imports a local
collision URDF, preserves its hand mounting frames, and combines it with the
saved object's collision pieces and finite tabletop. It runs position-level
kinematics/collision detection on CPU, never arm dynamics or policy training.

```bash
python -m real2sim2real.controller_actor_training.arm_path_audit \
  --config /data/arm-audit.json
python -m real2sim2real.controller_actor_training.arm_path_review \
  --audit /data/arm-audit/audit.json
```

The configuration pins `trial`, `urdf` and `sources`; declares a fresh `output`,
finite `deadline_s`, `base_body`, at most eight `placements`, up to 1000 samples
and an IK iteration budget. Every placement supplies `position_m` and `yaw_rad`
in the upright task frame. Table height relative to the robot base is an explicit
experimental assumption until measured. The imported URDF's fixed torso posture
is retained; placement search does not optimize torso joints.

Each `arms` entry gives a unique `name`, the fixture `source_root`, URDF hand
`root`, optional namespace `source_prefix`, ordered scalar hinge `joints`, all
arm `collision_bodies` including fixed wrist adapters, and `seed_rad`. Up to four
`initial_seeds_rad` may be declared. Multiple seeds are tried at the first
sample only; the lowest weighted position/orientation residual supplies the
continuous seed for later samples. This is local IK, not a globally complete or
collision-aware motion planner. Recorded wrist targets and clocks are unchanged.

The audit records:

- Position/orientation error and signed physical joint-limit margin per arm.
- Sampled joint-speed ratios against native URDF velocity limits. These are
  lower bounds on potentially higher unsampled speeds, not continuous proof.
- Arm/environment and arm/self penetrations. Contacts between joint neighbors
  through fixed mounting links remain in raw diagnostics but are separated from
  nonadjacent collisions. This classification uses topology, not observed
  collision magnitude; world geometry is never ignored. Two moving joints
  between bodies do not count as neighbors.
- Relative hand-body transforms after copying named finger joints. Matching
  root names alone does not establish mounting compatibility. FK agreement
  still does not certify collision geometry, masses or actuator equivalence.

`criteria` declares position error, angle error, joint margin and nonadjacent arm
penetration limits. A `screen_pass` requires both arms to meet all criteria and
the hand-frame comparison; it does not set `arm_integration_ready`. Actual arm
torque, dynamic contact, table/base collisions beyond the declared arm bodies,
continuous path clearance and robot sensing still need their own validation.
Failed local fits do not prove that no base, torso posture or grasp could work.

A different grasp family requires a fresh audit of its own saved wrist poses,
including acquisition, closure and lift. Reusing the old base positions can make
an otherwise reachable new grasp appear unreachable. Compare explicit independent
base placements and densely audit any passing shortlist. Payload sensitivity is
a separate physical gate; an accurate collision-free nominal IK path does not
erase a failed payload trial or establish loaded arm control.

Outputs retain all placements and sampled failures, native timing, original
states, asset hashes and implementation identity. The review adds per-phase
errors and top/side skeleton views at the largest recorded nonadjacent
penetration. These lines omit collision thickness and are labeled fitted IK,
not physical execution. The review itself has a two-minute CPU deadline.

The practical progression is to use arm feasibility **during grasp planning**:
consider both contact frames, elbow clearance, support-hand entry and base/torso
placement together. A successful Cartesian-wrist lift need not admit a physical
bimanual arm path. Preserve its object-motion target and physical criteria while
replanning the robot-specific grasp. See [grasp retention](GRASP_RETENTION.md)
and [testing](TESTING.md).
