# Complete translation candidates for downstream diagnostics

`reconstruction_repair.translation_candidate.prepare` turns a complete source
reconstruction plus a complete translation proposal into a separately hashed
diagnostic artifact. It supports taking a locally useful repair through
retargeting and a physical baseline while unresolved reconstruction checks stay
visible. Packaging does not approve a reference or start training.

```python
from real2sim2real.reconstruction_repair.translation_candidate import prepare

report = prepare(
    original_artifact, proposed_geometry_npz, new_output,
    evidence={
        "diagnostic_only": True,
        "clean_motion_used": False,
        "parent_artifact_sha256": parent_sha256,
        "geometry_sha256": proposal_sha256,
        "remaining_checks": ["metric calibration", "whole-clip continuity"],
    },
    hardlink=True,
)
```

The proposal must retain all source frames and geometry fields. Mesh vertices,
faces, scale, rotations, cameras and timestamps must be exactly unchanged; only
object translations can differ. Nonfinite values, changed shapes, missing frames
and evidence hashes belonging to another input are rejected. Human, masks, depth
and mesh assets are carried forward with content hashes. Hard linking is optional
and does not introduce a path dependency on the original directory.

The old quality and diagnostic files become **parent evidence**. Their scores
are not relabelled as measurements of the new proposal. The new candidate report
records the changed frame IDs and leaves metric validity, full-reference
approval, physical validation and actor-training approval false. It must not be
presented as a normally accepted repair or substituted into an active job.

Run explicit downstream checks and retain their receipts before granting any
later approval. A successful physics import or complete rollout can establish an
executable baseline while manipulation and reconstruction accuracy remain poor.
Short tracking prefixes are not supported by this handoff: do not fill missing
poses to manufacture a complete candidate.

Seven CPU checks cover immutable source preservation, parent-score scope, mesh,
timeline/rotation protection, mandatory diagnostic evidence and rejection of
missing/nonfinite original poses rather than silently filling them.
