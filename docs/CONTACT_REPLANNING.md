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
