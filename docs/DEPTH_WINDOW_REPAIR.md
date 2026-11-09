# Regenerated depth and anchored pose windows

Use this workflow when object tracking has good image overlap but conflicts
with predicted foreground depth. Static-floor agreement does not establish
small-object depth accuracy. Keep the original reconstruction and failed trials.

Declare the native frame interval, camera, centered crop, floor regions, anchor
frames and fit/held-out split before execution. The crop must remain centered
on the camera principal point; `foreground_depth.crop_camera` derives the
corresponding field of view. Review floor regions in RGB, exclude object/person
pixels, and check disjoint floor pixels. Full-image controls are observations
for diagnosis, not a per-frame selector based on a desired object pose.

`python -m real2sim2real.reconstruction_repair.crop_depth_sequence --request request.json`
accepts explicit export/depth/source-mapping/checkpoint/output paths, the crop,
reviewed floor regions and plane, full-image control IDs, a finite time budget,
and source-only provenance. It writes a native-window metric-depth archive and
per-frame qualification receipts. The derived archive records raw/aligned as
the same calibrated composite at identity scale. This does not describe the
original model's raw prediction. Original depth remains separately retained.
Invalid new pixels stay unobserved. Outside-crop integer pixels are restored
after encoding to avoid legacy float-to-millimetre truncation.

Revalidate anchors against the new depth. An old pose is a proposal for this
stage, even if it passed checks against a different prediction. A boundary pose
must also pass its join to the retained earlier reference. Independent endpoints
can initialize `anchored_recovery.refine`; a bounded additional interior fit-frame
anchor can help when endpoint interpolation misses the observed path. Every
new anchor requires actual renderer checks. Do not use held-out frames to choose
extra anchors or restored poses.

If sparse anchor interpolation remains a poor initialization,
`dense_anchored_recovery.refine` accepts additional pose proposals at declared
**fit frames only**. It initializes intermediate poses by translation
interpolation and rotational SLERP. Additional proposals are not locked anchors
and do not count as independent observations. Endpoint anchors remain exact.
The final acceptance comparison uses the original anchor-only hypothesis,
including image-preservation and displacement limits, rather than substituting
the warm start as an easier baseline. The helper rejects held-out initialization
IDs, duplicate proposals and changes to registered anchors. Callers must retain
how each fit-frame proposal was generated and still run actual GPU validation.

The optional `observation_preservation.preserve` step retains admissible original
fit-frame poses when a temporal fit regresses them. Silhouette score alone is
insufficient: the original pose must also pass depth, support and camera checks.
All window, displacement, anchor and motion checks run again. Acceleration may
change and must remain visible in the report. This is a CPU proposal step.

`python -m real2sim2real.reconstruction.validate_pose_sequence` checks proposed
poses against the native images, new depth, support and inter-frame limits with
the actual GPU renderer. It requires explicit source IDs, the window mapping
and the preceding pose. It stops at rejection and does not fit or carry poses.
Its receipt distinguishes a checked interval from a complete reconstructed
reference or physical execution.

Retain failed proposals and qualified disconnected windows separately. A longer
set of individually plausible observations is not necessarily a joined reference.
Diagnostic videos may show rejected intervals only with explicit uncertainty
labels. Never merge them silently into a selected reference or pair them with
a physical rollout made from different inputs. Complete packaging, retargeting,
scene stability and physical evaluation are subsequent gates.

GPU execution, temperature/power limits, deadlines and clock ownership belong
to the calling application. Software recovery may reuse a bounded lease;
scientific rejection must remain distinct from a startup failure. Numerical
source, model hashes, frame mappings and archive revisions accompany the run.
