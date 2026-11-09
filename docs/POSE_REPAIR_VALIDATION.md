# Pose proposal bounds and validation provenance

`reconstruction_repair.pose_bounds.project_rotation_bound` projects a rigid-pose
proposal into an explicit geodesic rotation-correction bound. It preserves
translations and leaves already-in-bound poses exact. A box constraint on three
rotation-vector components does not bound their vector magnitude; do not assume
an optimizer's component bounds establish the final angular correction limit.
Projection is not acceptance: recompute image, depth, support, temporal motion,
interaction and seam checks afterward. No defaults or recorded poses change
automatically.

`reconstruction.validate_pose_sequence` retains its default recovery-strength
validation for every proposed pose. A recorded normal tracker update can have
passed a different **existing** image threshold than a fresh recovery proposal.
An all-recovery audit can therefore reject an otherwise valid tracked update.
Keep that stronger audit result; do not describe it as a pass.

For an explicitly source-matched audit, provide `--tracking-evidence` with a JSON
manifest using schema `real2sim2real.tracked-pose-evidence/1`. Its `states`,
`updates` and `mapping` entries each contain a relative `path` and `sha256`.
The mapping must identify the same source video and consecutive native IDs.
Only exact, unchanged accepted `tracked_update` poses, with matching state/update
logs and no rejection, use the shared tracking gate. Recovered, modified or
uncovered poses keep the stricter recovery gate. Depth, support and motion
thresholds are unchanged, and every pose is rendered again. Results record the
mode for each source frame and the evidence hash; this is not an all-recovery
strength result or an actor-training approval.

An incomplete reference, a failed physical settling test, uncertain scale or
infeasible robot reach remain separate problems. Neither a renderer pass nor
geometric continuity establishes robot grasping or task success.
# Native frame-zero boundary

`validate_pose_sequence` accepts omitted motion history only when the proposed
native sequence starts at frame zero. It still checks that pose against RGB,
depth, support and camera validity, records `motion_history_checked=false` for
that one frame, and applies normal step checks from frame one onward. A later
window must provide its actual adjacent previous pose and source ID. Do not
invent a negative-frame predecessor to validate a new initial resting pose.
The focused boundary/provenance suite covers both cases.
