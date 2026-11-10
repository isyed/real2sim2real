# Articulated second-hand support

This CPU extension retains a successful object-centric primary-hand grasp and
adds a separately actuated hand beneath the lifted object's body. The object,
finite tabletop, collision decomposition, initial task state and primary contact
plan are preserved. Each hand has a force-limited Cartesian wrist fixture.
**No arm reachability, learned policy or hardware execution is established.**

## Inputs and stages

Start from a successful trial carrying its full `configuration`, `candidate`,
state archive hash and prepared scene. Pin `parent.trial` and
`parent.trial_sha256`. Use a fresh `output`. Preserve the parent `robot`, `task`,
`physics`, clearance feedback and orientation feedback settings.

Declare `support_robot` with an external flattened fixed-root MJCF, a name
`namespace`, the resulting namespaced `root_body` and ordered fingertip `sites`,
a distinct `fixture_prefix`, finger torque limit and wrist force/torque limits.
The source robot must declare radians. This composer supports mesh assets;
models requiring other asset kinds or external includes need an explicit adapter.
It namespaces classes/references, scopes global defaults, copies mesh/license
assets, appends six limited wrist joints and records hashes. It does not change
the object inertia or collision mesh. Meshes and licensed assets stay external.

Run each bounded stage as:

```bash
python -m real2sim2real.controller_actor_training.bimanual_grasp \
  --config experiment.json --phase prepare
```

Other phases are `plan`, `evaluate`, `validate`, `ablate` and `review`.
`deadlines_s` provides finite phase budgets; the supervisor hides the GPU and
limits CPU math threads. Every new stage output must have a fresh directory.

`support_planning` specifies `yaw_samples` (at most 12), `max_nfev` (at most 400),
`finger_offsets_m`, `seed_flexion_rad`, `approach_m` (at most 0.2 m) and
`opening_gap_m` (at most 0.01 m). The exercised four-finger layout uses anatomical
joint suffixes matching the external Sharpa assets. A different robot requires
reviewing its posture, site order, joint limits and finger geometry.

The planner takes the first hold sample of the pinned primary trial as a lifted
reference state. It copies named joint coordinates into the expanded model and
casts upward rays to the source mesh's underside. It fits the support fingertips,
checks insertion/seating at no more than 2 mm spacing, and checks the parked hand
against every retained pre-insertion baseline state. Hand–hand, hand–object,
hand–table and self contacts are included. Saved baseline samples do not certify
all intervening motion or changed-parameter cases; dynamic checks are required.

Contact IK uses position/collision calculations (`mj_fwdPosition`), not a force
solve on intersecting intermediate hypotheses. This avoids asking the dynamics
solver to resolve a rejected geometric candidate. Actual trials still use full
MuJoCo dynamics.

## Coordinated insertion and control

The primary hand first approaches, closes and lifts using its existing controller.
The support hand remains parked. It then approaches and seats under the object,
after which the declared hold interval begins. `support_controller.approach_s`
and `seat_s` extend the trial; they do not shorten the scored hold.

A single measured object translation aligns the insertion to its planned lifted
state, limited by `max_alignment_m` (at most 0.05 m). That alignment is then
frozen. Both wrists share subsequent changes in the primary vertical command.
The support hand does not follow a falling object's position downward. Rotational
and horizontal coordination are limited; there is no general bimanual planner
or arm collision solver in this adapter.

Optional `actuator_bias_compensation` offsets the support position commands by
robot gravity/Coriolis bias divided by actuator stiffness. It acts through the
existing bounded position actuators and retains their force limits. It never
sets external applied forces or compensates the object's gravity. This matters
when a geometrically clear support path otherwise sags into the tabletop.

Optional `load_feedback` supplies `target_weight_fraction`, `gain_m_per_n_s`,
`max_speed_m_s` (at most 0.02 m/s) and `max_offset_m` (at most 0.02 m). It integrates
a bounded upward wrist correction from **signed vertical force on the object**.
The target fraction multiplies the declared nominal mass and gravity; validation
mass changes do not reveal their true weight to this controller. Assessment uses
each trial's actual assumed weight. The integrator starts only after horizontal
insertion. Neither feedback option changes the primary grasp or actuator limits.

Object pose, contact force and robot dynamics are exact simulator observations
in this pilot. Deployment requires corresponding vision, force/tactile sensing,
state estimation and model assumptions. Do not describe this as a deployed
observation pipeline or learned policy.

## Evidence and acceptance

Primary clearance, opposed contact, wrist-frame drift, rotation and penetration
gates remain unchanged. Maximum robot penetration now includes **both** hands,
including hand–hand contacts. Second-hand acceptance additionally requires upward
force of at least 10% of actual assumed object weight in at least 80% of retained
hold samples, and at most 10 mm object-origin drift in the support wrist frame.
These are declared research criteria, not force-closure or hardware guarantees.

Contact wrenches use MuJoCo's contact-frame orientation and body-side sign.
Large lateral/normal contact force alone cannot pass the upward-load criterion.
Records include both hands' world forces on the object, the support torque about
object COM, individual support finger normal forces, both wrist poses, actuator
peaks and force-feedback offset. Planned fingertip targets do not imply loaded
fingers. Kinematic drift remains a proxy for in-hand motion, not a measured
contact-surface slip velocity. State/force records are sampled; penetration is
checked every integration step.

`evaluate` selects the lowest primary drift among fully passing nominal support
candidates, using the parent's closure. No pass means no selection. `validate`
pins that selection and freezes both controllers, contact plans and approach/lift
settings. It allows longer holds and explicit mass/friction cases. Later failures
are retained; no gate is relaxed or old outcome overwritten.

`ablate` uses the same pinned nominal selection, primary controller and scoring
window, with the support hand parked. `ablation.directory` and up to four explicit
`cases` declare this diagnostic. Support gates necessarily fail; the output cannot
promote a bimanual controller. Compare primary task metrics and object retention
to test whether the second hand actually helps.

Reviews replay retained states, include embedded WebM/GIF media and expose the
measured support rather than inferred target contact. Use `review_trials` for
comparisons across validation and ablation folders; `review_server` provides a
loopback browser alternative to an IDE preview.
