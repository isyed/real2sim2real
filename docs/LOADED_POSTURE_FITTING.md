# Loaded-snapshot finger posture fitting

`real2sim2real.retargeting.loaded_posture.fit_loaded_pad_offset` fits one shared
offset on explicitly named, owned hinge joints across saved configurations.
It uses position kinematics, pad Jacobians and collision witnesses. It does not
step physics, alter the input configurations or replace a controller.

Supply full model `qpos` snapshots and per-snapshot pad normal targets in world
coordinates. Omitted position targets preserve each original pad position.
Use measured or reconstructed contact geometry to justify target normals, and
label planned contact normals as hypotheses when local surface evidence is
unavailable. Keep each snapshot's source, time, force observations and role in
fitting versus validation in the downstream experiment.

The fit balances pad position, normal alignment, bounded joint change and
additional penetration. Native joint limits intersect across the snapshots and
optional `command_samples_rad` in the named-joint ordering. Its offset ceiling
cannot exceed 0.05 rad. `normal_regression_deg` optionally penalizes worsening
an individual fitting snapshot's alignment. This is a soft optimizer penalty,
not an acceptance guarantee: check the actual result on reserved snapshots.

`minimum_geom_separation` checks explicit geometry sets with MuJoCo signed
distances and witnesses. Distances above `cutoff_m` are reported as a lower
bound, not an exact measured clearance. Select physical obstacles separately
from planning exclusions. The helper does not change collision flags. Use the
embodiment's existing contact filtering for robot self-collision; excluded
adjacent links are not automatically forbidden physical contacts.

Review the entire hand, including proximal links and palm, and intermediate
correction fractions. Sampled fractions do not prove swept-volume clearance.
Report both absolute limits and changes from the original state. Loaded measured
joints may already lie slightly outside nominal ranges under soft constraints;
distinguish inherited violations from new ones and from command-limit checks.
An unchanged planning-exclusion overlap still remains an unresolved absolute
acceptance issue.

Keep the first fit frozen if validation motivates a tighter objective. Identify
reused development snapshots and reserve additional states for the revised fit.
Report worst-case regressions as well as mean gains. A tiny average alignment
improvement or better static clearance does not establish restored pressure,
contact retention, successful inversion or control readiness. Only an admitted
candidate should proceed to the unchanged loaded rollout comparison.

Tests: `tests/test_loaded_posture.py` checks ownership, conflicting native bounds,
input preservation without stepping physics, multi-snapshot alignment trade-offs,
and signed plane distances/cutoff semantics.
