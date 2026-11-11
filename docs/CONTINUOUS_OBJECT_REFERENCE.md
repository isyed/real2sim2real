# Continuous object-reference acquisition and motion

`controller_actor_training.continuous_reference` creates a kinematic command plan
on the capture clock. Robot approach and closure use the video's pre-lift interval.
The demonstrated object motion supplies lift and return. There is no separate
robot lift or restart of reference frame zero after acquisition.

The input is a hash-pinned live-finger arm baseline with admitted grasp geometry,
scene and object reference. A rejected motion baseline can supply geometry and
controller settings; its rejection remains. Output schema
`real2sim2real.continuous-reference-plan/1` has `physics_run: false` and
`success: false`. States are planning proxies, requiring a fresh dense arm audit
and actual free-object dynamics.

## Planning contract

Captured poses, frame IDs and timestamps remain pinned. Pose increments use the
capture-world axes and an explicit resting-object anchor, which can be the existing
gravity-settled task pose. This does not reproduce the capture's absolute starting
pose. `continuous_protocol` defines the new timing; donor reference settings
identify the source archive.

The reference runs from time zero. Original approach/closing finger waypoints
finish on that clock. Planning rejects translation exceeding 10 mm before
acquisition completion. Both wrists follow common object increments while
preserving their admitted relative contact arrangement and native limits.

If reconstruction noise puts the raw reference mesh below the table, planning
can raise its **command** within a declared budget of at most 10 mm. Each adjustment
is recorded. Scoring still uses the raw reference; no mesh, tabletop, capture pose
or threshold is edited. Exceeding the budget rejects planning.

Command sampling is 10–25 ms, within a 1,000-sample audit budget, with subsequent
controller interpolation. Native capture timing has no speed scaling. Terminal
settling adds a declared 1–2 seconds. The first route retains the grasp during
table return; release/retraction remains separate work.

```bash
python -m real2sim2real.controller_actor_training.continuous_reference \
  --config /data/continuous-plan.json
```

Required keys: `baseline_result`, `baseline_sha256`, new `output`,
`sample_period_s`, `terminal_settle_s`, `maximum_table_adjustment_m`, and
`deadline_s` (<=900). The bounded CPU CLI writes `protocol.json` and
`source-trial/{result.json,states.npz}`, plus an implementation snapshot.

Use module CLIs for this experimental preparation route. Pass the new plan and
its source hashes to `arm_path_audit`, retaining original collision/joint/speed
criteria and screening every timestamp. A passing audit can feed
`arm_replay_pilot`, which also checks the assembled finger/arm/environment path.
Planned object states are never written into dynamics after initialization.

## Evaluation contract

Continuous table-to-table and staged suspended-hold tasks have different cohorts.
Preserve both protocols and identify which one each result evaluates.

- `approach` and `close` end at declared acquisition completion.
- `lift` covers the remaining interval before suspension.
- `hold` covers reference mesh clearance above the existing task minimum
  (50 mm downstream). It can contain motion.
- `return` follows suspension; `settle` follows capture completion.

Raw pose error covers acquisition, lift and return under existing 20 mm/10 degree
limits. Clearance, opposing-contact and load-sharing gates apply to the
reference-defined suspended segment. Actual clearance does not choose that segment.

Both wrist-frame drifts anchor at the first actual saved sample at or after
acquisition completion and cover lift, carry, return and settling. Existing 10 mm
limits remain; these measurements are stricter than post-lift anchoring.
Arm and robot constraints retain their original limits.

New table-return requirements reject sampled mesh/table penetration above 3 mm.
Terminal settling needs table contact in >=80% of final samples, maximum final
clearance <=5 mm and the declared settling duration within 50 ms sampling tolerance.
These checks do not certify release or hardware placement.

`stationary_hold` follows acquisition and the demonstrated lift to the first
suspended sample, then freezes wrists and object target for the remaining clock.
It omits table-return evaluation. Existing staged inputs retain their old behavior.

Reference scoring includes acquisition/return and separates `complete_protocol`
from native-duration `moving_reference`; terminal settling cannot inflate the
latter. These use reconstructed references, not official challenge ground truth.

The scalar residual-training adapter rejects continuous results until its
observations, rewards and activation support acquisition anchoring and return.
Staged hold-only semantics cannot silently carry over to this new task.
