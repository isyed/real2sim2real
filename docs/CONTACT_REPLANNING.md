# Replanning support insertion and opposing pad placement

Use this stage after a retained grasp has demonstrated nominal physical success
but insertion or contact robustness remains poor. Keep the accepted controller,
source state archive, object trajectory, clock and physical acceptance criteria
as the comparison baseline. Planning results cannot replace a physical pass.

`controller_actor_training.contact_replan` generates bounded posture hypotheses:

- Rotate the support hand around a declared **world-frame contact pivot**. Keep
  its articulation and preserve the original world insertion/seating directions.
  Rotating the insertion displacement with the palm can sweep the fingertips
  into the object even when the final posture has more tabletop clearance.
- Increase the opening gap below the object without changing the final seating
  pose. This separates palm clearance from clearance beneath a lower-hanging
  payload. A tilted palm alone does not solve object interference.
- Apply explicit, bounded, named primary finger offsets as separate ablations.
  These are candidate commands, not newly observed contact positions.

The geometry screen evaluates the existing support controller against archived
object poses, including an 8 mm lower-object stress case. Alignment latches on
**time**, including the phase-boundary sample. The case is a geometric screening
assumption, not a mass-dependent prediction or a simulated payload experiment.
A conservative positive gap is measured above the tabletop's horizontal upper
envelope, using compiled collision mesh vertices or primitive support bounds.
This extends beyond the finite table footprint: negative clearance alone does
not prove collision. Actual finite-table contact witnesses remain separate.
Mesh-distance queries that return spurious zero distances for separated meshes
are unsuitable for certifying a positive clearance margin.

Run with a downstream JSON configuration:

```sh
python -m real2sim2real.controller_actor_training.contact_replan --config CONFIG.json
```

Required fields are `parent_trial`, `parent_trial_sha256`, `output`, `deadline_s`
(at most 900), `minimum_table_gap_m`, `fidelity_limits` (`position_m`,
`rotation_deg`) and up to eight `variants`. Optional `height_offsets_m` declares
one to four screening offsets between -20 mm and zero, including zero (default
`[0, -0.008]`). Each variant declares `name`,
`rotation_vector_world_rad`, `pivot_world_m`, optional `additional_opening_gap_m`
(up to 20 mm), and optional `joint_offsets_rad` (up to 0.35 rad per owned finger).
The baseline is replayed and its state archive must reproduce exactly. Rejected
geometry is retained. Admitted candidates receive fresh unassisted fixture
physics with unchanged controller settings and reference fidelity limits.
Each invocation uses a new output directory and preserves its implementation.

`controller_actor_training.loaded_finger_replan` additionally fits actual pad
points and normals against the reconstructed surface at a pinned loaded sample.
It reuses `retargeting.pad_contacts`, freezes the wrist and undeclared joints,
and exports bounded finger offsets for inspection. Required fields include the
pinned parent, `sample_index`, `maximum_offset_rad`, owned `joints`, `pads`
(`geom`, `outward_body`, and the parent's `target_index`), and `output`.

```sh
python -m real2sim2real.controller_actor_training.loaded_finger_replan --config CONFIG.json
```

The loaded wrist is re-expressed at zero fixture coordinates without moving any
world body. Soft-limit excess up to 0.01 rad may be projected into legal planning
limits; every projection is reported. Position, 25-degree normal alignment and
penetration checks remain explicit. Large residuals or offsets at their bounds
mean that a local fixed-wrist repair has not solved the contact geometry. Do not
silently transfer that result into control; reconsider the wrist/contact layout.

After a fixture pass, rebuild and densely audit **both arm paths**, then run the
unchanged arm controller. Freeze the new nominal candidate before matched
payload/friction trials. Compare identical frame intervals, timing, perturbations
and acceptance thresholds; an older shorter-motion robustness result is not a
matched control. Keep independent finger-force groups: more aggregate opposing
contact time does not establish broader load-bearing contact across fingers.

`arm_clearance_audit` now records both actual table penetration witnesses and
per-hand conservative positive-clearance minima by phase from saved arm states.
Neither this audit nor a successful IK path infers contact forces or reruns physics.

## Coupled wrist and pad hypotheses

`loaded_finger_replan` accepts optional `wrist_search` with `translation_m` (up to
30 mm) and `rotation_rad` (up to 0.35 rad **per fixture axis**). These bounds
intersect native joint limits. Declared finger changes may extend to 0.7 rad for
this coupled search; the fixed-wrist default retains its 0.35 rad bound. These
are search ranges, not changed actuator limits or acceptance tolerances.

A pad can use `target_mode: simulated_contact_anchor` to select its deepest
admissible object contact in the saved pose. The reconstructed surface must lie
within 3 mm of the collision witness. The contact point is expressed in the
owning body frame and projected onto its actual compiled convex collision hull
within 1 mm, with a compatible surface normal. This supports curved thumb pads
and palm bearing patches whose active contact lies away from a center ray.
It is privileged simulation geometry, **not independent observation or measured
force**. No object contact may be substituted with a table or self-contact.

`retargeting.pad_contacts.pad_frame(..., point_body_m=...)` supports the same
explicit surface patch. The original center-ray behavior remains the default.
`pad_grasp_planning` preserves explicit patch points through subsequent fitting;
it must not silently recreate the patch from a ray through the pad center.

Optional `target_shift_world_m` proposes a bounded contact-layout translation
(up to 30 mm); shifted targets are projected to the reconstructed surface within
5 mm. These are new geometry proposals, not evidence of existing contact.

`controller_actor_training.coupled_grasp_transfer` takes pinned `parent_trial`
and `loaded_fit` paths/hashes. The parent must be a successful retained fixture
trial; the fit must pass its geometry checks. It transports the fitted wrist
back to the settled initial task using world-from-body matrices:

`T_world_object_initial @ inv(T_world_object_loaded) @ T_world_wrist_loaded`.
The object-relative grasp is preserved; no loaded object state is copied into
physics. It then fits opening and bounded preload paths, optionally searches up
to eight declared approach offsets (150 mm maximum), and checks collision,
position, normal alignment and static gravity balance before a fresh rollout.
`preloads_m` supplies up to three values; `opening_gap_m`,
`maximum_normal_force_n`, `fidelity_limits`, and `output` are explicit.

```sh
python -m real2sim2real.controller_actor_training.coupled_grasp_transfer --config CONFIG.json
```

The static force budget is a declared per-contact planning assumption; failure
under it does not prove that torque-limited hardware cannot hold the object.
Finite contact-patch reassessment can reuse `object_grasp_pilot --phase pad-patches`:
samples on one pad must share its budget. A position-error pass alone does not
establish a bearing patch. Missing near-surface samples remain rejected, even
when an aggregate IK residual looks small. New pad plans use their explicit
closing path rather than adding the legacy undirected flexion offset; physical
acceptance criteria and actuator limits remain unchanged.
