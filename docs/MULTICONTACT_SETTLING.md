# Flat-object contact diagnostics

A reconstructed object that rocks before robot contact can expose a solver
contact-manifold issue, as well as a geometry or pose error. Isolate these causes
before changing the reconstruction. Compare a known analytic box on the ground
and on the actual mesh support, then compare the unchanged reconstructed object.
Keep support transforms, collision geometry, friction, mass, inertia, timestep
and solver settings identical within each comparison.

Newton's MuJoCo solver exposes `enable_multiccd`. In the tested installed version,
its default is false. Single-contact convex pairs can poorly represent a flat
surface resting on another surface. Multiple-contact CCD can generate up to four
contacts per pair. It is an explicit numerical experiment, not an object-pose
repair or guarantee of stable grasping. Positive geometry margins and backend
support can affect its operation; inspect the installed solver and measured
contacts. See the [Newton solver documentation](https://newton-physics.github.io/newton/stable/api/_generated/newton.solvers.SolverMuJoCo.html).

The public runtime now exposes `SimConfig(enable_multiccd=True)` and forwards it
to solver construction. Its default remains false, preserving existing actors
and experiments. For a resolved controller configuration, set the same explicit
boolean in both `sim.enable_multiccd` and `env.sim.enable_multiccd` before
packaging a new artifact. Preserve the previous policy and evaluation receipts.
This changes the simulated dynamics; do not silently apply it to a running
learner or claim an old actor/rollout was produced with the new mode.

The isolated tester supports matched comparisons:

```sh
CUDA_VISIBLE_DEVICES='' python -m real2sim2real.controller_actor_training.isolated_object_stability \
  --urdf object.urdf --pose initial_pose.json --support support.usda \
  --output multicontact_cpu --device cpu --seconds 10 --impratio 1 \
  --enable-multiccd
```

Repeat without `--enable-multiccd`, using a separate output. Set `--impratio` to
the target controller's value; the tester preserves its historical default of
20. Results record both settings. Test long enough to expose late sliding:
a short transient pass can fail a ten-second rest interval. Small initial-height
and orientation perturbations are diagnostics, not adopted reference edits.

Matched CPU controls can identify a contact-mode effect without deforming an
otherwise plausible collision mesh. They do not establish GPU behavior, robot
balance, grasping, task success or reconstruction accuracy. Keep measured
scenario results in the downstream experiment record. Confirm on the intended backend and then perform
unassisted full-controller evaluations with recorded configuration and producer
identity. Changing physics requires fresh evaluation rather than relabelling
saved trajectories.

Separate object-rest improvement from robot-task improvement. Full-controller
tracking AUC can increase while the robot still falls at the same time, records
no task contact and has zero task success. Report fall timing, monitored contact
scope and success rates alongside pose error. Such a result can justify keeping
an explicit contact-mode candidate for further work, but does not by itself
qualify the reference for actor training or establish that reconstruction is
correct. State whether metrics use the reconstructed reference or independent
evaluation targets, and retain trial-count differences in comparisons.
