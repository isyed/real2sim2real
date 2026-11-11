# Opening and insertion geometry before grasp optimization

`retargeting.insertion_geometry` and
`controller_actor_training.handle_access_audit` test idealized tool envelopes
against the prepared object, its convex collision proxy and the finite tabletop.
They do not fit hands or run dynamics. A visible opening can admit a straight
probe while leaving no room to hook beneath its rim.

The audit reuses `object_grasp_pilot.verify_scene_assets`, now available without
initializing MuJoCo, and `load_plan` to reject changed assets or initialization.
It reuses `support_planning.underside_targets` to identify the first underside
intersection, avoiding cavity/top surfaces as substitutes for underbody support.
The source mesh, object pose and finite table remain fixed.

`vertical_grid` finds enclosed empty regions in a sampled vertical projection.
The mesh raster is a discovery aid; it can miss subpixel triangles and is not a
collision certificate. `union_columns` checks vertical lines against the union
of the convex proxy pieces, preserving overlapping-piece occupancy. It reports
where a proxy intrudes into the source opening, which global surface-error
statistics may hide.

Probes are swept balls; a rigid L-shaped hook or support bar is represented by a
union of capsules. Tool-centerline and translation-path samples carry explicit
cover-radius allowances. Distance to a mesh surface is 1-Lipschitz, so subtracting
both cover radii and a safety margin yields a conservative clearance lower bound
for this specified envelope against that fixed mesh. Signed solid membership
rejects samples inside the object. For overlapping convex pieces, inside means
inside *any* piece; parity over a concatenated overlapping mesh is not used.

The audit tests straight insertion, lowering an L-shaped tool then moving its toe
under a selected rim target, and horizontal insertion of a broad capsule support
bar beneath the object COM. Only the largest discovered opening is searched.
These are bounded templates, not an exhaustive articulated motion planner. Passing
does not validate a Sharpa finger, palm, wrist, arm, bearing surface, contact force
or lift. Other inner-wall grips, articulated curling and reorientation may differ.

At an underside target above a horizontal table, a circular probe of radius `r`
needs at least `2*r + 2*margin` of vertical space. The reported additional gap is a
geometric lower bound; robot shape, uncertain geometry and the motion needed to
create that gap are not included. Geometry-inflation sensitivity can reject a
nominal envelope pass without silently altering the source meshes.

```bash
python -m real2sim2real.controller_actor_training.handle_access_audit --config access.json
python -m real2sim2real.controller_actor_training.handle_access_review \
  --audit output/audit.json --reconstruction reconstruction \
  --output review/review.html
```

The configuration pins `source_config`, `source_config_sha256` and `plan_sha256`;
uses a fresh `output` and finite `deadline_s`; and specifies `grid` (`pitch_m`,
`padding_m`, `maximum_cells`), `minimum_opening_area_m2`, `radii_m`,
`depth_levels_m`, `headings` (4 or 8), `margin_m`, `tool_step_m`, `path_step_m`,
`shank_length_m`, `toe_length_m`, `engagement_overlap_m`, `bar_length_m` and
`inflation_m`. Lengths are metres in the prepared upright world. CPU execution
hides CUDA and preserves previous outputs.

The browser review shows source/proxy opening maps, clearance witnesses, a metric
side section and optional capture projections. Capture overlays are visual checks,
not independent verification of hidden bearing geometry. Tests cover annular
openings, overlapping convex solids, table collisions and obstacles between path
samples. Existing MuJoCo scene checks continue through the shared asset verifier.

## Reusable modules and external alternatives

| Component | Appropriate role |
|---|---|
| `grasp_kinematics`, `pad_contacts`, `pad_grasp_planning.path_audit` | Fit actual collision pads and test full-hand paths once an insertion strategy is geometrically plausible. |
| `support_planning`, `bimanual_grasp`, `support_control` | Underbody support, joint two-hand execution and wrist coordination. The current underbody planner requires a pinned lifted state; it cannot establish initial tabletop access. |
| `contact_proxy`, `verify_scene_assets` | Preserve collision/source correspondence and reproducible scene assumptions. |
| Do as I Do `pipeline/solve_ik.py` | Mink wrist/fingertip target fitting. Needs an adapter for object-centric planned targets; demonstration inputs are not an opening planner. |
| Do as I Do `pipeline/optimize_physics.py` | Sampling-based MuJoCo-Warp MPC after a feasible nominal sequence. Preserve external acceptance checks and distinguish assistance/warmup from unassisted success. |
| Do as I Do deployment replay | Later arm placement and collision-aware dual-UR3e IK; not a remedy for blocked finger insertion. |

The external inspection used local upstream revision
`824591b808c342b20079c3b4198a8c2bdf88c74e`. The local checkout is sparse; relevant
retargeting source blobs were read from Git, and the deployment role is documented
in its README. No external implementation was copied or executed. Its defaults
include different penetration, warmup and optional contact-guidance behavior, so a
full pipeline substitution would not be a matched comparison without adaptation.
Retain upstream licenses for any future integration; do not import neighboring
repositories at runtime. Reusing selected components is preferable to changing
the reconstruction and control stack to address a geometric clearance failure.
