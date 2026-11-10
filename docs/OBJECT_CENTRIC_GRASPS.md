# Object-centric grasp pilots

The [reusable learning index](REUSABLE_LEARNINGS.md) maps reconstruction/control
findings to code, regression tests and remaining capabilities.

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

## Feedback and robustness experiments

Optional `contact_feedback` integrates an extra, bounded closure command for the
opposing fingers from their summed normal contact force. Its config declares
`target_force_n`, `gain_rad_per_n_s`, `max_speed_rad_s` and `max_offset_rad`.
Optional `orientation_feedback` uses observed object orientation error and the
wrist rotational Jacobian to command bounded wrist corrections. It declares
`gain_s_inv`, `max_speed_rad_s` and `max_offset_rad`. Both modify robot joint
targets under the existing joint/force limits. They never modify object poses,
geometry, forces applied directly to the object, or human reconstruction.

The pilot presently reads exact simulated object orientation and contact forces.
Those are privileged observations until a corresponding estimator/sensor path
exists. Force groups distinguish thumb, individual opposing fingers and remaining
hand geometry. A five-fingertip geometric plan does not imply that all five
fingers carry load in the actual grasp.

`grasp_robustness --config study.json` pins a previously successful grasp, compares
up to 24 controller/case combinations and conditionally validates a frozen
selection. Controller variations are restricted to the two feedback options;
scene, contact targets and success thresholds remain fixed. `--validate` accepts
a `robustness_validation` config with comparison/protocol hashes, controller name,
finite deadline, explicit cases and hold duration. Diagnostic continuation of a
failed development candidate cannot promote it, even if later cases pass.

Trial schema 2 measures `hold_slip_m` in the **moving wrist frame**:
`p_relative = R_world_from_wrist.T @ (p_object - p_wrist)`.
The maximum displacement from the first hold sample must remain within 10 mm.
Schema 1 used the changing world vector `p_object - p_wrist`; intentional wrist
rotation could therefore be confused with slip. Schema 2 retains that quantity
as `hold_relative_world_displacement_m`. This is a coordinate correction, not a
larger tolerance. `object_grasp_audit.audit` can rescore saved hold kinematics,
checking source/state identity and reproducing recorded positions, without
overwriting historical verdicts or claiming new physical evidence. Translation
drift remains a proxy for in-hand motion, not measured surface slip.

`object_grasp_review.review_trials` creates a saved-state comparison of explicitly
selected trials. Reviews provide VP8 WebM, the original MP4 and an embedded
animated GIF fallback for local viewers. `real2sim2real.review_media` can also
prepare portable media from an existing video, verifying complete decoding and
source/cache hashes. This changes display files only; it never reruns physics.

The primary WebM can be embedded directly in HTML for IDE previews that do not
resolve sibling video paths. `python -m real2sim2real.review_server --directory
/path/to/review-root --port 8765` provides an alternative browser URL on
`http://127.0.0.1:8765/`, with HTTP byte ranges for seeking and no stale caching.
It serves review/media formats only, rejects hidden/outside-root resources, and
expires after eight hours by default (`--seconds`, maximum 24 hours). It does not
publish the report externally. A working browser check does not replace checking
the user's IDE preview.

`grasp_hold.measure_hold` and `hold_rejections` centralize the measured
hold criteria used by physical trials. `contact_support` summarizes observed
normal force and active sample fractions separately for each named hand group.
These summaries report actual support rather than inferring it from planned sites.

## Object tasks versus demonstration repair

In demonstration repair, the unknown is what the human actually did. Corrections
need support from RGB, calibrated depth, visible joint identity and other
observations. Making a simulated robot succeed does not independently establish
the truth of the repaired human fingers or contact timing.

In an object task, the goal is declared directly: for example, lift a pot clear
of the table, keep it upright and hold it. Robot contact locations, approach,
joint commands and feedback may be optimized against that goal. Human motion
can suggest a handle or approach side as a low-confidence proposal prior. It
does not constrain the robot to reproduce uncertain finger positions. Unknown
human articulation can remain unknown while robot manipulation is investigated.

Independent joint evidence can include held-out, anatomically identified image
landmarks, calibrated views not used for fitting, or motion-capture measurements.
Another prediction from the same fitted model is not independent evidence.
Appearance tracks need semantic identity checks: a tracked skin point is not
automatically a particular joint. A secondary view reused in fitting/depth is
a consistency check unless a clear evaluation holdout is maintained.

Independent contact evidence can include visible touching/contact transitions,
multi-view separation constraints, or measured tactile/force signals. Images
often constrain contact without proving its force or exact location. Simulated
contact forces test the consequences of a hypothesized robot scene; they are
not observations of the human's original contact.

Physics begins with free-body settling and proceeds to actuated contact trials.
Policy training additionally needs a task-specific environment adapter: resets,
bounded actions, attainable observations, reward, success/termination rules and
evaluation cases. A programmed feedback controller is not a learned policy.
Exploratory learning can address remaining control failures, but cannot certify
incorrect geometry, unobservable deployment inputs or infeasible arm motions.
Before substantial training, demonstrate a feasible sustained grasp, validate
the simulator/reset/observation boundary, and freeze a separate evaluation set.
An arm-based policy further requires the intended arm and its constraints.
