# Review observations before pose optimization

`reconstruction_repair.foreground_visibility.artifact_audit` records native
foreground masks, visible depth counts, depth spread and gaps across a complete
source timeline. `reconstruction_repair.observation_plan.plan(audit)` routes this
evidence toward mask review, depth/visibility review or further gap inspection.
It changes no geometry and grants no reference approval.

Small nonempty masks can still select background. Inspect native RGB overlays
at the beginning, during manipulation and after the object moves. Distinguish
real occlusion from a segmentation track that stays behind on a table or floor.
Numerical image-overlap scores against the wrong mask cannot establish target
tracking.

After RGB inspection, pass an optional `semantic_review` containing the audit's
`artifact_sha256` and `video_sha256`, boolean `passed`, source
`reviewed_frame_ids` and textual `evidence`. A failed review takes precedence
over available depth and explicitly blocks pose optimization. A review bound to
different source hashes is rejected. Passing sampled review is still not a
whole-clip tracking, metric calibration or physical-execution approval.

For a failed target review, repair segmentation first, then reassess every
dependent stage. A mesh generated from an incomplete mask may also need fresh
generation and metric calibration. Preserve the original complete bundle and
all rejected hypotheses; never fill observation gaps with invented motion.
