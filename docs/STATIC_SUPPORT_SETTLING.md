# Static supports and initial-rest diagnostics

An isolated settling test has no controller or reset loop to hold a kinematic
support in place. A support represented as a free-root kinematic body can acquire
backend-internal motion even when the Newton state copied for display remains
fixed. Inspect the actual model topology, not only the reported support poses.

The isolated tester now defaults to `support_mode="world_static"`. The existing
USD parser supplies the same shapes, scale, transforms and material settings;
`static_support.add_world_static_support` transfers only those shapes into the
world. No support body, free joint or support DOF is created. The simulator's
ordinary kinematic support API remains available for environments with their
own explicit reset/pinning lifecycle.

```sh
python -m real2sim2real.controller_actor_training.isolated_object_stability \
  --urdf object.urdf --pose initial_pose.json --support support.usda \
  --output isolated_cpu --device cpu --seconds 3
```

Hide CUDA before a CPU-only test. The historical
`real2sim2real.controller_actor_training.object_stability` entry point remains
compatible. `--support-mode kinematic_legacy` reproduces the earlier support
representation; keep that result labelled with its mode. The standalone
`isolated_object_stability` and `static_support` modules also work with older
pinned runtime support parsers, allowing an explicit diagnostic extension
without changing a running learner's numerical environment.

The result reports backend, support mode, shape/body/joint counts, initial
motion and late drift. `assess_settling` evaluates explicit caller-visible
limits. A passing CPU test does not certify the GPU backend or task execution.
Changing the tester must not erase an earlier failure or silently relabel its
result as a pass.

For flat objects that rock on mesh supports, use the controlled
[multiple-contact diagnostic](MULTICONTACT_SETTLING.md) before deforming their
collision mesh. Both contact mode and `impratio` are explicit, recorded options;
the historical defaults remain unchanged.

Static support alone cannot correct an unstable object pose. When reviewed
source frames demonstrate an initial resting phase, the optional
`reconstruction_repair.support_seed.resting_orientation_hypotheses` enumerates
nearby mesh-derived support orientations whose projected centre is inside a
convex support face. It preserves the world centre and does not snap the object
onto the table. These are geometric candidates, not physical-stability proofs.
The companion
`reconstruction_repair.support_seed.local_rest_orientation` proposes a bounded
rotation about a mesh-local centre, with explicit hold/fade frame IDs. Later
poses remain byte-for-byte unchanged. A hypothetical resting orientation must
still pass source-image/depth gates, transition checks, fresh retargeting,
packaging, repeated settling and unassisted physical rollouts. The helper never
widens contact thresholds, changes grasp labels, certifies training readiness,
or edits its inputs. Frame-window evidence and acceptance belong to the caller.

For any accepted initial-pose repair, rebuild the paired mesh/body-frame
registration and reference together. Verify corresponding world-space points,
initial object-body pose and support identity after packaging; a changed mesh
must not accidentally reuse the previous inertia or controller provenance.

## Flat-contact solver controls

A static table can still expose contact-generation artifacts. Before modifying
the reconstructed object again, compare an analytic primitive and a mesh box on
the ground and on the same table, with identical pose, material and timestep.
In a bounded CPU diagnostic, both boxes were stable on the ground but rocked on
the mesh table. Explicit multi-contact generation stabilized those controls and
the unchanged reconstructed object. This identifies a solver-setting hypothesis;
it is not evidence that every source pose or reconstructed base is correct.

`SimConfig.enable_multiccd` and the isolated tester's `--enable-multiccd` flag
enable this experiment explicitly. Their default remains false. The flag must be
a boolean; every result records it. `solver_ablation.package` creates a separate
frozen-policy artifact differing only in that setting, with unchanged geometry,
trajectory, inertia, controller and source artifact. It never mutates an active
learner or certifies the new policy. Recheck the intended GPU backend and perform
new unassisted robot trials before adoption. Preserve off/on controls and earlier
failures; improved contact numerics do not establish a grasp or task completion.

Match the solver ratio as well as the backend when interpreting a failure.
The follow-up controls at the robot policy's ratio admitted both off/on modes on
CPU and GPU, with less drift when multi-contact was enabled. The much larger
earlier disabled CPU failure used a different ratio; it is not a pure backend
comparison. Full robot trials showed a small tracking-score gain but no grasp,
with a balance failure before the predicted hand phases. Preserve those outcomes
separately from isolated settling, and avoid changing fingers or collision
geometry merely to compensate for an already fallen body.
