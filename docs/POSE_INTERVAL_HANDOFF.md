# Complete pose candidates from measured intervals

`reconstruction_repair.pose_candidate.compose` overlays explicit consecutive
native source IDs on a complete reconstruction geometry. Supply rigid
camera-frame poses, a named source hash and its original gate warnings for each
replacement. Overlays never interpolate missing frames, change human/mesh/camera
fields or rescale geometry. The output ledger records the source owning each
frame, every changed-source join and all motion-limit failures.

The ledger separately counts `replacement_frames` and `parent_fallback_frames`.
A complete stored timeline can still contain unqualified original fallback
poses. `failed_join_frame_ids` and per-join source names expose incompatibility
between individually accepted intervals and retained parent poses. Full-motion
failures inherited inside a parent interval remain separate from new failed
joins. Neither storage completeness nor independently accepted segments imply
that the resulting complete reference passes motion checks.

A recorded failed gate requires `explicit_warning_acceptance=True` in its
evidence. This preserves the failure; it does not relabel the gate as passed or
approve a complete reference. The application owns which warnings may be
accepted and the reason. Native image/depth/support/motion validation and
physical evaluation are separate evidence.

Individually valid intervals can have incompatible joins, particularly after a
depth revision changes absolute camera distance. Inspect the whole composed
timeline. A contiguous connecting interval from the same proposal can avoid
mixing incompatible old/new endpoints, but it must receive fresh native checks
including the preceding seam. Do not smooth or blend the join merely to make
its movement report pass.

Retain per-interval depth provenance. A pose previously measured against the
original depth is not automatically validated against regenerated depth. Keeping
that measured prefix and validating a new tail against its own declared depth
does not repair a rejected depth prediction in an unused variant. Keep both
archives, native frame mappings, hashes and their distinct roles explicit.

`pose_candidate.prepare` packages a complete pose-only candidate with the
unchanged paired human, mesh, camera and timeline. Parent quality/diagnostics
are stored under separate parent roles. They cannot become candidate-quality
measurements just because packaging succeeded. Optional named extra files can
retain interval ledgers and alternative validation-depth archives.

After nomination, independently validate the changed span, retarget the paired
reference and run an unassisted controller. Physical trajectories must come from
that exact packaged reference and evaluator. Keep former fallbacks and all
failed proposals; changing a preferred nomination must not rewrite historical
results or alter an already running job's input snapshot.
