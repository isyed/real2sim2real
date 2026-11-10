# Simultaneous body and rim grasps

The optional bilateral route creates a two-hand grasp before the initial lift.
It complements the existing lift-then-insert support controller. It is intended
for reconstructed body/handle-root regions where sharing load closer to the
center of mass may reduce a single-hand torque lever.

Planning and actual retention are separate prerequisites. A contact fit and a
static wrench solution cannot establish that position-controlled fingers will
produce or maintain those forces in physics.

## Planning

```sh
python -m real2sim2real.controller_actor_training.bilateral_grasp --config CONFIG.json
```

The downstream configuration pins a successful `parent_trial` and its SHA-256,
provides `output`, `maximum_normal_force_n`, `search`, and up to twelve explicit
`proposals`. Existing scene, table, gravity, mass, friction, robot and actuator
limits are verified through the parent's prepared scene. No reference geometry
or hand trajectory is required.

Each proposal declares `angle_rad`, `height_m` and one radial-ray contact family:

- `body_cup`: external body surfaces for both hands.
- `rim_pinch`: thumb inside, fingers outside the wall.
- `rim_outer_thumb`: thumb outside, fingers inside the wall.

Source-mesh intersections and normals select actual surfaces. An absent inner
wall, unsuitable normal or target outside the declared body radius rejects the
proposal. This is a bounded family search, not exhaustive grasp synthesis.
Current hand adaptation uses configured right/left anatomical pad names; other
embodiments need an appropriate naming/contact adapter.

`search` declares finite IK budgets, `seed_flexion_rad`, `hand_long_axis_body`,
`palm_axis_body`, optional `thumb_outward_body`, metric contact radius/tangential
span, opening gap, preload and approach distance. Hand axes must be explicit,
orthogonal unit vectors. Do not infer wrist orientation solely from the mean
normal of already curled distal pads: this can initialize palms inside a vessel.

Both hands receive equal fixture translation/rotation bounds. Tightening
`position_scale_m` changes the IK residual weighting, not the physical limits or
admission thresholds. The two complete hands are then checked together along
synchronous approach/closing paths with bounded translation and angular sampling.
Finite contact patches use the existing 4 mm sampling radius, 0.5 mm surface-gap
and opposing-normal checks. Samples on one pad share one normal-force budget.
Combined static load balance uses `retargeting.paired_contacts`.

`bilateral_refine` optionally refines a pinned pair with mutual collisions in the
same optimization. The pad fitter accepts two disjoint hand roots, corresponding
fixture prefixes, and up to ten pad frames. Its position/normal Jacobians and
collision derivatives include both hands. Failure to preserve contact patches
or normal alignment after separation remains a rejection.

## Closing, lifting and physics

```sh
python -m real2sim2real.controller_actor_training.bilateral_evaluate --config CONFIG.json
```

Evaluation takes a pinned plan, up to four admitted `candidate_ids`, explicit
`fidelity_limits` and `output`. It requires paired geometry/force admission, then
re-centers fixture coordinates at the closed wrists. Every waypoint is checked
by FK for an unchanged world pose. The interpolated path is checked again since
changing Euler coordinates can change interpolation between endpoints. No joint
limit is enlarged and no loaded object pose is copied into physics.

`support_controller.mode: simultaneous_grasp` makes both hands approach and close
on the primary rollout clock. Both close before lift. The second hand follows
the primary's world vertical command, including clearance correction. There is
no extra support insertion phase. The default `lift_then_insert` mode retains
its prior behavior and timing.

`controller_profile: inherited` retains the parent's feedback. The optional
`rigid_bilateral` profile removes independent primary orientation/handle-retention
and second-hand vertical load corrections; the pair shares lift and demonstrated
rigid object motion. This is an explicit controller comparison with the same
actuator limits and physical acceptance criteria. It is not force regulation.

The default `motion_condition: demonstrated` evaluates the existing reconstructed
object trajectory and native increment clock after the grasp. An explicit
`stationary_diagnostic` instead tests a fixed-world lift/hold. Its physical result
is labeled separately and **cannot pass demonstrated-motion admission**, even
if the stationary task succeeds.

Always retain failed rollouts, solver/clock health, source hashes and contact
forces. Advance a new grasp to arm dynamics only after a complete physical and
reference pass; test payload/friction and timestep sensitivity before promotion.
