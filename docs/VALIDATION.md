# Extraction validation

Validation date: 2026-10-29. No GPU workload, model download or cloud instance
was started for these checks.

| Check | Result |
| --- | --- |
| Portable repair regression suite | 125 passed |
| Core/stage adapter suite, including actual numeric repair and subprocess resume | 19 additional tests passed |
| Native retargeting-to-FlashCHORD Parquet relocation | Passed separately in the Torch retargeting environment; skipped in the thin CPU environment |
| Source syntax and sibling-workspace import audit | Passed |
| Core plus four module wheels | All five built; source/config/assets-code payloads checked |
| Wheel imports from a neutral directory | Checked without a source checkout on the import path |

The checks include invalid stage chains, overwritten outputs, artifact hash
changes, invalid deadlines, shared depth/camera reprojection, training-budget
rounding, protected Hydra overrides, early assistance removal, native human
payload preservation and object paths after relocating a robot reference.
They exposed and fixed partition metadata loss when flattening a Parquet
artifact.

Not yet established by this extraction:

- A fresh build of the combined reconstruction image with selected upstream
  GPU backend commits and all licensed model assets.
- A full new RGB clip through the combined reconstruction/repair/retargeting
  adapter and subsequent physical simulation.
- Performance, memory use, grasp quality or training throughput of the new
  package layout on a particular GPU.

The code is a modular extraction with CPU-verified handoffs. Its original
backend implementations have been used in prior research workflows, but that
history is not a substitute for validating this new combined entry point.

## Isolated reconstruction workers

The subsequent dispatcher update passes 26 core/adapter CPU tests, with the native
Torch-dependent interop test skipped in the thin environment. Eight new tests
cover argument-array container prefixes, malformed commands, changed input
rejection, stopping on a failed phase, refusal to overwrite partial work, and
importing the dispatcher without Torch/NumPy/PyAV/trimesh. CPU-only offline imports
also pass in existing separate segmentation, mesh-generation and tracking
containers. Video decoding was exercised separately. These checks do not yet
establish full inference in the new phase layout.


The more complete worker check found a missing SAM3D Body source package named
`data` (transforms and batch preparation, not dataset assets). Those six files
are restored with original headers and hashes. All six full model CLI imports
now pass in offline containers with GPU access disabled. A rebuilt reconstruction
wheel was checked for the worker and restored transform/batch modules. The repair
suite currently passes 126 tests; the adapter suite passes 26 with one optional
Torch interop skip. Full reconstruction acceptance is in progress, not yet passed.

The first full execution completed RGB segmentation, MoGe2 depth, direct human
motion and human-aligned depth. It stopped when mesh generation selected the wrong
Torch model cache. The cache-routing fix passes its CPU regression check; both
required DINO weights load offline on CPU from the corrected cache. Mesh/tracking
and complete-artifact validation subsequently passed, as recorded below. The original run
and completed intermediate predictions are retained by the downstream caller.

The offline dependency-check path is also tested across all six model phases:
it requests only CLI help, hides CUDA and creates no inference output directory.
This prevents a preflight from leaving an empty model-output directory that could
block the actual run. The focused dispatcher/backend suite passes 15 tests.


## Full reconstruction acceptance after integration fixes

Two complete reconstruction artifacts have now passed schema/hash checks. One
resumed from verified completed segmentation/human/depth predictions after fixing
model-cache selection; the second exercised the entire fresh RGB-to-artifact
chain. All source frames were retained (858 and 719 respectively). The module's
point-map checks passed and the same aligned-depth hash was recorded for mesh
generation and tracking. Runtime used separate offline model containers; this
does not establish a fresh build of the optional combined image.

The caller retained original outputs and generated full-source-timeline overlays
for review. Both regenerated objects have markedly better silhouette agreement
and initial floor placement than their original outputs. These comparisons do
not isolate a single causal algorithm change or establish absolute metric scale.
Temporal jumps and human/floor alignment warnings remain. No reconstruction
repair, robot retargeting, physics or actor training was run in this acceptance.
Those stages remain separate validation requirements. Detailed scenario results
and artifact identities are held by the downstream project.

The execution fixes promoted during acceptance are phase-isolated workers,
complete human preprocessing source packaging, per-model cache routing, and a
full-backend offline dependency-check mode without inference side effects.
No evidence-dependent resize, smoothing or human-position correction has been
silently promoted into reconstruction defaults.


The optional full-human-surface decoder passes axis/unit/world-transform and
invalid-input tests. A real licensed MHR model was decoded on CPU for two retained
frames, producing finite 18,439-vertex surfaces and 36,874-triangle topology. This
is mesh decoding, not a new prediction or retargeting test. The focused core,
module-contract, pipeline, dispatcher and surface suite passes 35 tests. Complete
new comparison videos using this surface path are verified by decoding every
encoded frame; downstream receipts distinguish completed videos from pending
renders and record the source geometry and human-surface hashes.


The quality-report tests verify that successful tracker flags cannot hide poor
image agreement or motion jumps, missing scores remain unassessed, invalid pose
and time arrays fail, and good scores do not grant physical/readiness approval.
The focused core/module/dispatcher/surface/quality suite passes 39 tests. The
quality role was added after the active run's source freeze; its newly integrated
publication path has not yet completed a fresh GPU campaign. Retrospective
reports must be kept separate from historical published artifact manifests.


Three additional fresh RGB-to-artifact executions completed with source-frame
counts of 778, 896 and 708. Their source snapshots and input/output hashes are
retained by the downstream caller. Matched retained-reference comparisons use
identical masks and sample frames rather than stale tracker scores from a
replaced trajectory. All three fresh outputs have worse sampled silhouette
agreement than the corresponding retained parents; successful execution is
therefore not a claim of improved reconstruction. No additional physical
execution or actor training is established by these runs.

The optional resting-support seed search passes three synthetic tests: bounded
parent-preserving adjustment, rejection when the required movement exceeds the
budget, and refusal without reviewed resting-phase evidence. The existing
adapter/metric-preflight/seed suite also passes 17 tests. A real predicted-data
seed improved its declared initial resting clearance while retaining parent
image/depth gates; fresh tracking is pending and no physical result is claimed.

The reusable CPU comparison renderer passed a three-frame synthetic execution:
complete MP4 decoding, frame count, cache reuse, and rejection after surface bytes
changed. It records source video, geometry and human surface hashes. Full new
repair comparison videos remain downstream jobs, separate from this smoke test.

Explicit tracking-runtime and dispatcher checks pass 16 tests. The object-motion
backend now verifies both score/refiner model files even for its offline CLI
preflight. A separate offline CPU check loaded both real configurations and
checkpoints (116 and 134 tensors); this does not validate GPU kernels or tracking.

The repair suite passes 13 tests after adding numeric paired hand-separation
reporting. Its regression test verifies that a rejected outline improvement
records an observed increase above the unchanged 0.005 m limit. No acceptance
threshold changed. A real standalone GPU tracker subsequently completed a full
clip with explicit checkpoints; reconstruction quality remains separately gated.

Native prompt-review and reconstruction-dispatch tests pass 22 checks. They cover
native-image bounds, degenerate/nonfinite boxes, source-frame range, later-frame
preview limitations and the distinction between valid coordinates and correct
subject selection. The new first-frame preview does not automatically recognize
objects or approve prompt semantics. A downstream retry validates the actual
corrected segmentation/depth behavior separately.

Six additional fresh reconstructions completed with all 4659 source frames and
hash-validated artifacts, including the integrated quality-publication path.
Their immutable inference snapshot predates the prompt-preview addition; that
addition has separate CPU validation. One RGB-only prompt correction enabled a
previously failed human-depth alignment to complete all 546 frames, with at least
12180 corresponding pixels per frame under the unchanged 100-pixel minimum.
Completed execution still exposed substantial tracking/registration issues; none
of these results establishes retargeting, physical task success or actor readiness.

Metric-calibration/preflight checks pass 25 tests after adding conflicting-frame
reporting and correcting rejected-calibration guidance. Held-out evidence still
cannot fit the scale; rejection still produces no calibrated mesh. The new test
checks exact conflicting frames, unchanged fit-versus-held-out acceptance, and
review text that preserves the input mesh on rejection. No metric threshold or
normal fine-repair requirement changed.

The experimental depth-evidence helper passes seven synthetic tests covering
protected-pixel preservation, zero/missing-only changes, unchanged unsupported
and clean frames, inconsistent/unreviewed anchors, nonfinite data and invalid
limits. The related metric calibration/preflight suite still passes 25 tests.
A downstream run verified all 546 raw depth payloads, every unaffected aligned
payload, and unaffected calibration arrays remained identical. It retained the
same twelve calibration samples/split while qualifying contradictory pixels in
26 frames. Native seed validation subsequently prevented reference promotion;
no new full tracking, physics or actor-training result is claimed.

The extracted numerical recovery/RGB adapter passes 13 CPU checks, including
geometric vetoes despite good silhouettes, the stricter recovery image threshold,
no support snapping, actual rollback after a bad update, use of the RGB refiner
flag, and unaccepted failure-state records. A separate offline container check
imports the real exporter/new adapter and loads both actual tracker configurations
and checkpoints (116/134 finite tensors). This does not validate GPU tracking;
the new full-clip recovery acceptance trial remains pending its guarded lease.

Recovery-source binding adds three CPU regression checks: the selected engine
outranks the container working directory, an already-cached foreign exporter is
rejected, and an incompatible observed-seed API is rejected before tracking. The
combined recovery/RGB/binding suite passes 16 tests. Corrected offline preflight
prints the actual frozen exporter source and loads both real checkpoints; this
still does not replace the guarded GPU acceptance run.

The recovery suite passes 17 tests after adding exact proposal diagnostics and a
500-configuration float32 motion-bound regression test. The generator leaves a
small representational margin inside the existing 0.08 m limit. This fixes a
numerical proposal defect; it does not establish full-clip reconstruction quality.

The guarded downstream integration trial completed 285 accepted updates before
rejecting the next frame of a 546-frame clip. Image/depth and temporal constraints
had no jointly acceptable candidate. The numerical fix is CPU-validated, but this
full-clip RGB recovery trial failed: no complete repaired reference, robot physics
or actor-training readiness is established. The original artifact was retained.
