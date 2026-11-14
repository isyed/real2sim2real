# Retargeting

For small finger corrections around saved loaded configurations, see
[loaded-snapshot posture fitting](../../docs/LOADED_POSTURE_FITTING.md). It fits
owned joint offsets and supports independent whole-hand clearance checks before
any loaded rollout; it does not certify dynamic grasp retention.

`object_grasp` and `grasp_kinematics` provide a separate object-centric path:
surface contact proposals and bounded hand IK from calibrated scene geometry,
without a human finger reference. See the
[grasp pilot workflow](../../docs/OBJECT_CENTRIC_GRASPS.md) for supported hand
conventions, optional CPU MuJoCo dependencies and physical validation limits.

This is the actual constrained IK retargeter, not a learned motion generator.
It converts native human MHR predictions to SOMA while retaining per-frame
identity, then uses Pink/Pinocchio tasks and robot joint limits to fit the
robot's body, hands and root to that human/object reference.

The first included robot configuration is G1. Supply its complete URDF/mesh
asset closure and SOMA/MHR models externally. `configs/g1` is a template;
`asset://...` paths resolve under the explicit `assets` directory. To add a
robot, supply the frame-alignment and IK mapping JSON files and a matching
robot URDF. Retargeting support does not by itself supply a trained body
controller for a different robot.

Input source time is preserved. The current exporter rejects nonuniform frame
times rather than inventing a playback speed. It reports the floor alignment
source; an explicit `support_plane` is preferred over its body-derived floor
fallback. An optional `support_asset` must already be transformed to the
final robot frame and contain no external USD references.

Output is a portable `motion_v1` Parquet with object mesh/URDF closure and a
hand–object proximity schedule. The default 2 cm contact threshold labels
reference proximity; it is not evidence that the simulated fingers touched
or held the object. Physics evaluation follows retargeting.

`options.soma_assets` can point to a separate local SOMA model directory when
body-model assets and robot assets are provisioned independently. The standalone
stage creates its output directory and refuses existing retargeting results.
`retargeting.frames.reframe_object` applies rigid world/body changes to the
paired mesh and trajectory without scaling or changing their world surface.
OBJ export omits vertex colors/textures for predictable collision import while
preserving geometry; this does not simplify the simulation mesh.

`retargeting.segmented_support.component_hull` builds one explicitly masked,
seeded depth component at a time. It rejects disconnected background and
enforces caller-supplied depth, height and metric-span bounds. Package separate
support parts separately; a single hull over a broad rectangular scene selection
can fill floor/background space and gaps below a table.

For a planar support, `planar_component_hull` fits a robust top with a declared
normal/tilt prior. Closed-slab thickness comes from observed lower-side points;
without sufficient side evidence it fails rather than adding arbitrary padding.
This is a collision approximation of predicted depth. Validate image agreement,
initial object placement and robot clearance, then evaluate physics before
claiming task improvement. Numerical routines require no private dataset or
neighbouring checkout.

Support extraction now defaults to `static_geometry(..., mode="surface")`.
It fits an above-floor horizontal surface from the caller's RGB selection and
predicted depth, excludes human/object pixels, follows a seeded connected plane,
and measures thickness from adjacent observed side points. It rejects a flat
surface without thickness evidence rather than adding padding. The automatic
result recovers the observed surface only; it does not invent legs or a base.
Use `segmented_support` for separately selected visible structural parts and
`surface_support.validate_components` before policy packaging. Keep reviewed
scene-specific masks/seeds in the downstream project. The historical filled
whole-region hull is available only through explicit `mode="legacy_convex"`;
it can join unrelated background points and must not be silently reused.

Before training a hand-closing controller, use
`retargeting.reference_contact.audit(vertices, faces, object_poses, hand_points)`
to check whether the reconstructed hands can approach the represented surface.
Pass object-to-world rigid poses `[T,4,4]`, world-space hand samples `[T,H,P,3]`,
explicit side names and a threshold in metres. The function performs exact
triangle-distance tests near the threshold; sparse mesh vertices alone cannot
establish lack of surface proximity. It reports per-hand frame IDs and distance
bounds without changing geometry, motion or labels. No proximity can indicate
missing handles or other thin grasp features, human/object misalignment, or a
task without hand contact. Inspect source evidence before changing labels or
thresholds. Geometric proximity does not establish physical contact or readiness
for actor training. A bounded search fails explicitly if its budget is exceeded.

The same audit accepts foot or other task-specific samples through explicit
`sides` labels such as `("left_foot", "right_foot")`. Returned `effectors` are
independent of the legacy `hands` field. Inspect the demonstrated interaction
first: a successful foot push does not require hand-contact phases, and missing
hand proximity alone is not a reason to reject its reference.

`insertion_geometry` discovers projected openings, checks convex-union occupancy
and bounds swept-sphere clearance with explicit sampling allowances. These are
geometry diagnostics, not whole-robot or physical grasp certificates. See
[HANDLE_INSERTION.md](../../docs/HANDLE_INSERTION.md) for the prepared-scene audit.

For completed surfaces, `conditional_hand_scene` builds explicitly conditional
kinematic scenes, initializes wrist seeds from actual pad frames, and checks
sampled paths using all enabled hand collision geometry, including palm and
proximal links. See [surface envelopes and full-hand fitting](../../docs/CONDITIONAL_HAND_ENVELOPES.md).
These checks do not admit observed geometry or establish loaded retention.

`articulated_acquisition` adds explicit mirrored target ordering, bounded
preshape/insertion waypoint construction and full-path candidate checking.
See [articulated acquisition](../../docs/ARTICULATED_ACQUISITION.md) for convention
audits, unused-finger posture and complete obstacle restoration before holds.

`contact_uncertainty` adds convex-pad sampling, explicit tapered-wall support
evidence and a fail-closed geometric closure-stop diagnostic. See
[contact support versus clearance](../../docs/CONTACT_UNCERTAINTY.md). These
checks distinguish a separated pad from a well-seated grasp and preserve the
boundary between hypothetical kinematics and force/retention validation.

`CollisionReserve` can add selected-link collision costs from aligned uncertainty
scenes to pad fitting while preserving all primary and evaluation collisions.
Bounded normal-residual weighting helps diagnose seating/clearance tradeoffs.
See [contact uncertainty](../../docs/CONTACT_UNCERTAINTY.md) for full-path gates,
default compatibility, and the distinction between geometric support and hold
validation.

`collision_visualization.CollisionRenderer` caches enabled convex collision
surfaces for CPU-only kinematic/saved-state reviews. It never steps the simulator
or changes collision geometry. Label animations by their evidence type and verify
encoded frames; playback alone is not a physical rollout or retention result.

`loaded_contact_repair.propose` generates bounded wrist/finger changes while
retaining pad frames from an actual loaded snapshot. It does not certify force
retention; replay the full acquisition and hold. See
[loaded contact repair](../../docs/LOADED_CONTACT_REPAIR.md).

Hollow-container collision hypotheses use separate convex walls and a bottom,
preserving the opening for inversion and placement. See
[nominal arm and scene diagnostics](../../docs/NOMINAL_TASK_DIAGNOSTICS.md).
