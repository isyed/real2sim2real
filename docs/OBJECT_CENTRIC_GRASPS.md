# Object-centric grasp pilots

This path plans robot contacts from a reconstructed object and finite tabletop,
then evaluates a short grasp/lift/hold in CPU MuJoCo. Human finger trajectories
are not control targets. Optional uncertain position cues can influence proposal
ranking; they cannot override geometric or physical admission checks.

The current pilot supports an articulated hand on a six-joint, force-limited
Cartesian wrist fixture. **It does not include a robot arm or establish arm
reachability, whole-body balance, hardware execution or full-task success.**
The default pose/closure convention uses anatomical joint suffixes from the
Sharpa hand assets. Adding a different hand requires reviewing that mapping,
site order, geometry/contact names, actuator limits and initial posture.

Install the optional `grasp` extra of `real2sim2real-controller-actor-training`
alongside the matching core, retargeting and egocentric modules. This adds
MuJoCo, CoACD and CPU geometry/video dependencies; models, robot XML/meshes and
private captures remain external. The exercised runtime uses MuJoCo 3.8.1 and
CoACD 1.0.11. Existing training/GPU runtimes are not modified by this workflow.

## Scene boundary

`python -m real2sim2real.egocentric.object_task --config task.json` exports:

- A copy of the metric reconstructed object, with its original body frame.
- Explicit `world_from_object` poses, frame IDs and capture timestamps.
- The accepted upright support polygon and declared collision thickness.
- An explicit task goal, such as lift height, minimum clearance and hold time.

Its config supplies `reconstruction`, `support`, `frame_start`/`frame_count`, a
fresh `output` and `goal`. The support's geometry hash must match reconstruction.
No hand files, clean reference motion or reference scans are read. The output
`scene.json` is a generic metric `real2sim2real.object-task/1` boundary. Object
reference motion supplies context; it does not force the simulated object.
Use an outer finite deadline for this lightweight preparation command.

## Bounded stages

Run each stage through the supervised CPU entry point:

```bash
python -m real2sim2real.controller_actor_training.object_grasp_pilot \
  --config experiment.json --phase prepare
```

Other phases are `plan`, `approach`, `evaluate`, `validate` and `review`.
`deadlines_s` declares each invoked phase's maximum runtime. The parent process
hides the GPU, limits CPU math threads and terminates descendants on failure or
timeout. Each output directory must be fresh; keep failed experiments intact.

The experiment config declares:

- `task`, `output`, and optional separate `planning_directory`,
  `evaluation_directory` and `review_directory` names.
- `robot`: a flattened, fixed-root MJCF hand with its mesh closure, `root_body`,
  ordered thumb/finger `sites`, finger torque and wrist force/torque limits.
- `physics`: nominal `mass_kg`, `friction`, and `timestep_s`. These are assumptions,
  not quantities inferred from RGB. Inertia uses uniform density over the closed
  source mesh at the supplied mass. Prepared physics and actuator limits cannot
  silently change when loading the scene.
- `collision`: decimation size, bounded CoACD options/piece count and sampled
  surface-error limits. Failed closure/topology checks must be investigated.
- `planning`: grasp-width/height limits, proposal/IK budgets, contact-error and
  penetration limits; optional `finger_offsets_m` and `orient_finger_order` define
  contacts spread along a represented handle or other narrow feature.
- `rollout`: approach, closure, lift and hold durations; approach offset, closure
  correction and lift displacement. `closure_search_rad` optionally evaluates
  up to three offsets, without changing admission limits.
- Optional `clearance_feedback`: bounded gain, raise/lower speeds and wrist
  travel. It adjusts the robot wrist using the object's measured simulated
  clearance. A real robot would need an observation source for this feedback.

Preparation copies the robot assets with their available notices, constructs a
convex decomposition of the object and a finite tabletop slab, and saves hashes
and surface-distance diagnostics. Mesh contacts use convex representations in
MuJoCo, so one convex hull can erase a pot's cavity or a handle's opening; sampled
surface checks do not fully certify topology. See the
[MuJoCo mesh reference](https://mujoco.readthedocs.io/en/stable/XMLreference.html#asset-mesh).
The source visual mesh and nominal mesh-derived inertia remain separate from
collision pieces. CPU videos simplify display meshes only; they preserve a
closed object's watertightness when simplifying it.

## Planning and simulation

`retargeting.object_grasp.surface_pairs` samples opposing outer-surface rays in
principal horizontal directions, checks width and approximate antipodality, and
requires all requested neighboring finger targets to lie on the source surface.
This is a bounded proposal generator, not a global grasp-optimality or
force-closure certificate. Sites must be ordered consistently with finger offsets.

`retargeting.grasp_kinematics.fit_contacts` fits scalar robot joints and bounded
wrist corrections using analytic MuJoCo site/contact Jacobians, joint limits,
collision witnesses and a weak initial-posture prior. It includes robot self,
object and tabletop penetration. The `approach` phase pins and rechecks a saved
contact plan, then searches nine wrist translation directions at no more than
2 mm sample spacing. It never changes the contact IK or object pose. A sampled
collision-free path still requires dynamic execution.

Before planning, gravity settling checks rest velocity, displacement, orientation
and support clearance. Default `initialization: reconstruction_preserving` limits
displacement to 20 mm. Explicit `gravity_settled_task` instead defines a **new
task start**, with a 50 mm displacement bound and the same rotation/rest checks.
The report preserves both poses and whether the original 20 mm check passed.
Choosing this mode does not repair or replace the reconstructed demonstration.

The controller approaches, closes, lifts and holds. With clearance feedback,
wrist motion responds to actual object clearance rather than assuming wrist
height equals object lift. Only robot joints are actuated. The loader rejects
object actuators, non-free objects, gravity compensation on the object, object
springs, equality constraints, mocap bodies and changed gravity. Runtime checks
reject external body/joint forces. Position actuators remain force limited.

Evaluation retains all outcomes. The current research gates require the declared
hold clearance, thumb/opposing-finger contact in at least 80% of hold samples,
less than 10 mm relative hold drift, less than 30 degrees object rotation, and
the declared penetration tolerance. Records include actual contact forces,
per-actuator peak force/torque with units, table contact, physical states and
controller commands. These are explicit pilot criteria, not hardware guarantees.
MuJoCo's [soft-contact model](https://mujoco.readthedocs.io/en/stable/computation/index.html#soft-contact-model)
and uncertain material properties limit interpretation.

A nominal pass selects a candidate for declared mass/friction cases; no passing
candidate means no selection. `validate` pins a successful parent evaluation and
tests the same grasp/closure on up to twelve explicit cases. It does not reselect
or retune the candidate. A deterministic repeat is a reproducibility check, not
an independent estimate of success probability.

Execution rejects plans whose scene receipt or settled start changed. Validation
also checks the selected result against its pinned evaluation, contact plan and
saved state hash. Changing a task, grasp or controller requires a new experiment.

The review loads saved states without rerunning dynamics, renders CPU videos,
fully decodes them, and binds each video to its state hash. It shows initialization
changes and representative passing/failing trials. Detailed tables retain all
results. Nominal lift success does not validate uncertain human fingers, a full
episode reference or a physical robot.
