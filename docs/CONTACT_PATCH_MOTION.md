# Loaded contact motion and bounded patch replanning

`contact_patch_audit` replays a pinned grasp case with the same candidate,
controller, payload, friction, clock and duration. It adds read-only telemetry at
the runner's existing coherent `mj_forward` sample states. Source/result/state
hashes and implementation snapshots are preserved. Require an identical state
archive before interpreting the diagnostic; it cannot promote a failed source.

```bash
python -m real2sim2real.controller_actor_training.contact_patch_audit \
  --config /data/contact-audit.json
```

The downstream config gives `source_result`, `source_sha256`, `output` and
`diagnostic_settings`: `minimum_force_n`, `sliding_speed_m_s`,
`rolling_speed_rad_s`, and `maximum_sample_gap_s`. The last bounds integration
across consecutive loaded samples. These are diagnostic thresholds, distinct
from the source's physical/reference acceptance criteria.

For every named pad, the adapter uses solved object contact forces, world contact
points, and both bodies' point/angular Jacobians with their current velocities.
It records tangential material speed, relative tangential angular speed,
object-local and pad-local force-weighted contact centroids, collision-piece IDs,
and a linear tangential-force/friction ratio. Other hand/object contact geoms are
logged separately so planned pads cannot hide load elsewhere.

Tangential material velocity provides sampled evidence of sliding. Low sliding
speed with relative angular motion is **rolling-compatible**, not proof of a
persistent material correspondence. Patch centroids can jump as a convex contact
manifold changes: their migration is never relabeled as sliding distance. The
linear friction ratio does not account for all torsional/rolling constraint
dimensions and is not a full friction-cone certificate. These are simulator
observations, not independent contact annotations or hardware measurements.
For the underlying force/torque dimensions, see the
[MuJoCo contact model](https://mujoco.readthedocs.io/en/stable/computation/index.html#contact).

## Replanning a pad surface and posture

```bash
python -m real2sim2real.controller_actor_training.bilateral_patch_replan \
  --config /data/patch-replan.json
```

`bilateral_patch_replan` takes a pinned admitted bilateral plan and an audit
that exactly reproduces a trial of that plan. One to four proposals give new
body-local patch-ray directions per hand/anatomical group. Each ray change is
bounded to 35 degrees on the existing compiled collision hull. Finger posture
changes are bounded to 0.35 rad and native limits; both wrist poses and other
fingers are frozen to numerical tolerance during the static fit.

The stage refits grasp/closed coordinates while preserving original open and
pregrasp waypoints. It then rechecks the full pair's approach/closing path,
normal/position errors, finite pad patches and shared-budget gravity balance.
Missing contacts or collision failures remain rejected. A new ray is a geometric
proposal, not an observation of the human's contact or new collision geometry.

The resulting plan can use `bilateral_evaluate` for fresh acquisition, complete
stationary hold and frozen demonstrated motion. `grasp_sensitivity` repeats the
original declared physical cases only after nominal admission. Static fitting,
improved slip statistics or a passing reference metric cannot erase a failed
physical grasp. Loaded arm dynamics still need their own validation.

A new fixed posture may recover one load and lose another. Do not pool case
successes from different source plans into a complete matrix pass. The sensitivity
stage freezes one admitted candidate/controller for every case. A policy that
selects or transitions between postures from supported-load observations needs
its own pinned rule and fresh full-matrix validation, including the transition.


## Bounded contact placement as well as patch-ray changes

Optional `surface_shifts_m` maps each hand role and anatomical pad group to a
world-frame contact proposal within 6 mm. The stage projects onto the unchanged
source mesh within 3 mm and preserves the intended wall side by requiring a
nearby surface normal within 35 degrees of the original. A global nearest point
can jump across a thin vessel wall; a wrong-side projection must not silently
replace an intended opposing contact. Proposed surfaces still need every original
normal, finite-patch, shared-force and paired-path admission check.

`seed_offsets_rad` gives named finger-only optimization seed changes of at most
0.2 rad inside the existing 0.35 rad search radius and native limits. It does not
expand physical limits or move frozen wrists. Replanned grasp/closed errors and
finite patches are refreshed in the resulting hand metadata. Original pregrasp
and opening waypoints stay fixed. All proposals, including rejected wrong-side
or incomplete patches, remain explicit downstream evidence.

For fresh arm-mounted stationary and demonstrated evaluation, use the admitted
`arm_replay_pilot.finger_plan` binding described in
[arm integration](ARM_HAND_INTEGRATION.md#stationary-acquisition-diagnostics-and-admitted-finger-replanning).
A complete stationary hold pins one controller/clock before a motion retry.
