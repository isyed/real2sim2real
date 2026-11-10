# Articulated arms with retained grasp hands

An accepted Cartesian-wrist grasp does not establish an arm-executable grasp.
Use separate admission steps: assembled-model checks, coordinated path checks,
bounded actuator-driven physics, and only then policy training. Keep hand joint
commands, source timestamps, task geometry and physical acceptance criteria
explicit so a change of embodiment remains reviewable.

## Reusable components

* `arm_hand_assembly.build_assembly` attaches existing hand subtrees to independently
  placed fixed-base MJCF arms. It removes only explicitly enumerated root fixture
  joints and their actuators. Finger joints, collision geometry, actuator settings
  and task objects remain in the source scene. Assets are absolute and hashed;
  generated scenes and robot meshes belong in downstream artifact storage.
* `extract_mounted_arm` separates an upstream arm and adapter from an included
  hand, recording the original mount transform. It handles MuJoCo's Euler
  convention explicitly; SciPy lowercase `xyz` is not a substitute for MuJoCo
  lowercase `xyz`. Unsupported include/constraint structures fail closed.
* `ur_arm_model.prepare` expands an explicitly supplied official UR description
  using optional Xacro, then imports its collision URDF into MuJoCo. Record and
  justify joint-limit overrides. A vendor parameter file can differ from a newer
  hardware manual; do not silently substitute a nearby arm variant.
* `arm_path_audit` supports one shared base or named independent bases. It checks
  the original wrist poses, relative hand-body FK, joint margins, sampled speed
  limits and nonadjacent arm collisions. Up to eight placements and 1,000 source
  samples are allowed within a required process deadline. When a hand remains a
  fixture, copy its joint axes as well as its root pose and label the partial
  integration. It cannot establish complete two-arm control.
* `arm_replay_pilot` requires a passing two-arm audit covering every archived
  timestamp. It removes both fixtures and drives arm joints with torque-limited
  position actuators, bias compensation and optional velocity feedforward. Finger
  commands replay the archived controller outputs. Only initialization writes the
  free object's pose; no equality, mocap, external force or repeated object-state
  assignment assists the task. Arm speed, joint margin and collision checks supplement
  the existing grasp/support/reference checks. The explicit-PD helper is available
  for diagnostics, but the runner uses MuJoCo implicit actuator damping.
* `arm_replay_review` renders saved physics states on CPU, displaying arm collision
  meshes where no visual meshes exist. It reuses portable WebM/MP4/GIF playback,
  including embedded WebM for restricted IDE previews. It does not rerun dynamics.

## Configuration boundaries

`assembly.arms` declares the MJCF path, namespace, source hand root, removed fixture
joint names, mount body and rigid `mount_from_hand` transform. Optional adapter
cylinders and visual-mesh collision proxies are assumptions recorded in a receipt.
They are not verified manufacturing geometry. `arms` declares arm joint names,
collision bodies, source/target hand prefixes and IK seeds. `placements[].bases`
maps each independent base body to `position_m` and `yaw_rad` in the existing world.

A dynamic pilot pins the audit hash and passing placement, declares proportional
and damping gains in Nm/rad and Nm·s/rad, and specifies every arm joint's torque
limit. The underlying task's clearance, slip, opposed contact, load-sharing and
penetration gates are reused. The initial reference gates are 20 mm and 10 degrees,
matching the guided-grasp pilot. This is a diagnostic transfer of saved commands,
not the original online object-feedback controller and not a trained policy.
The old fixture's Cartesian force/torque limits are not equivalent to the new
arm's joint torque limits; do not describe these as identical controllers.

```bash
python -m real2sim2real.controller_actor_training.arm_path_audit --config /private/path/audit.json
python -m real2sim2real.controller_actor_training.arm_replay_pilot --config /private/path/pilot.json
python -m real2sim2real.controller_actor_training.arm_replay_review --directory /private/path/pilot-output
```

## What an integration result means

Vendor flange compatibility, a demonstrated hardware combination, an exact CAD
assembly, an IK pass and a physical rollout pass are different evidence. Keep
them separate in the report. Independent bases improve placement freedom but
their stands, cables and unobserved environmental obstacles also need modelling
before hardware claims. Sampled collision tests do not certify continuous clearance.

When IK passes but contact fails, inspect actuator saturation, wrist lag and
load sharing before changing hardware or training a policy. Explicit high-gain
velocity torques can oscillate on low-inertia wrists; implicit actuator damping
is a different, separately logged controller. Velocity feedforward is an ablation,
not an assumed improvement. Retain unsuccessful trials and original thresholds.

Public integration references: [Sharpa flange CAD](https://github.com/sharpa-robotics/sharpa-opencad-library),
[Franka/right-Sharpa assembly](https://tml.stanford.edu/cloak/setup/sharpa.html),
[official UR descriptions](https://github.com/UniversalRobots/Universal_Robots_ROS2_Description).
External models and their notices stay with their downloaded assets; this project's
Apache license does not replace the licenses of those assets.
