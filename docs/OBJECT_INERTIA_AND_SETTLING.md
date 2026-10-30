# Reconstructed-object inertia and settling

A mesh's coordinate origin need not be its physical center of mass. Assigning
every reconstructed object a center of mass at that origin and a fixed inertia
tensor can make a correctly placed object tip or move before a robot touches it.

The retargeting exporter derives center of mass and rotational inertia from the
closed metric mesh using `retargeting.object_inertia`. Its default nominal mass
remains 1 kg; uniform density over the reconstructed volume is an explicit
assumption, not measured material or real-world mass. `write_urdf(...,mass_kg=...)`
accepts another supplied mass. The mesh, link frame and reference poses are
preserved. Open or inconsistently wound geometry requires an explicit modelling
decision; it does not silently receive a placeholder or convex-hull inertia.
An `.inertia.json` receipt records the assumptions and geometry/URDF hashes.

Test the exported body alone before interpreting an actor's failure:

```bash
CUDA_VISIBLE_DEVICES='' python -m real2sim2real.controller_actor_training.object_stability \
  --urdf /data/object.urdf --pose /data/pose.json --output /data/cpu-settling
```

The pose JSON contains `pose_xyzw`, a seven-element world pose. The probe imports
the runtime's convex collision hulls and uses its nominal 200 Hz solver settings,
without a robot, tracking force or artificial assistance. CPU MuJoCo results are
diagnostics. `--device cuda:0` uses MuJoCo-Warp and must run under the caller's
configured thermal and time guard. The probe is limited to 30 simulated seconds.
Set `FLASH_CHORD_CACHE_DIR` and `WARP_CACHE_PATH` to explicit writable caches.

Use matched before/after tests with unchanged geometry, contact settings and
initial pose. Good isolated settling is necessary for an initially resting task
object, but does not establish contact, grasping, task success or policy quality.
Follow it with unassisted robot rollouts and independent repeats. Preserve the
original policies and actors: an inertial change requires fresh evaluation and
may require retraining a policy that learned against the earlier model.
