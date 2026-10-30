# Module learning register

Record reusable findings here. Scenario identities, private videos, measured
results and deployment settings belong in the downstream experiment repository.
An implementation change is not evidence of improved reconstruction or robot
performance until the corresponding acceptance test has run.

| ID | Finding | Module change | Evidence and state |
| --- | --- | --- | --- |
| R-001 | Independently installed perception models may require incompatible environments. Selecting subprocess interpreters only for the GPU calls still leaves heavy export/packing imports in the parent environment. | Reconstruction now dispatches every phase, including packing, metric input preparation, object export and artifact publication, to an explicit worker environment. `backend_commands` accepts argument arrays for externally supervised container wrappers; `backend_pythons` remains supported. | CPU dispatcher tests and offline imports in separate segmentation, mesh and tracking environments pass. Acceptance passed on a verified full-clip continuation and a second fresh end-to-end clip, including all-frame tracking and artifact hash validation. |
| R-002 | A partially completed model chain must not be confused with a complete reconstruction. | Record the executing phase, argv, start/finish times, failure type and per-phase logs. Stop at the first failure; refuse to overwrite a partial output directory. | Tested with a failing depth worker: later phases never start, and reuse of the partial directory is rejected. |
| R-003 | Shared depth before mesh generation removes one inconsistent input path, but monocular scale remains a prediction. | Preserve the common camera, native depth grid and point-map validation in reconstruction. Compare independently the resulting size, image agreement, motion continuity, support and human proximity before selecting repair or retargeting. | Camera/depth round-trip and mismatched-camera rejection are CPU tested. Generalization across additional complete videos remains pending. |

For future entries, include the failure mechanism, minimal reusable change,
regression test, measured acceptance evidence, limitations and affected module.
Keep rejected ideas visible rather than promoting them to defaults. A successful
reconstruction check does not establish contact, physical stability or readiness
for actor training.

**R-004 — extraction completeness:** excluding every directory named `data`
removed SAM3D Body's transforms and batch-preparation Python code. Restore that
six-file package with its original license headers and provenance. Full backend
CLI import checks (not just wrapper imports) are now part of acceptance. A data
storage exclusion must distinguish source packages from datasets. Validation:
all six full backend CLI imports now pass offline with GPU access hidden. The rebuilt reconstruction wheel contains the restored package.


**R-005 — tracker validity is not reconstruction quality:** saved raw predictions
can mark most frames as tracker-reliable while their projected mesh overlaps the
observed object very poorly. Keep runtime validity separate from independent
image/depth, metric-size, support and continuity evidence. Downstream acceptance
reports now retain each measure separately. Do not select a reconstruction from
`tracking_reliable_frames` alone. No new automatic acceptance threshold has been
promoted into the reconstruction algorithm from this observation.

**R-006 — model-family cache isolation:** a global human-model `TORCH_HOME`
selected a DINO repository without the object model's required pretrained weights.
An offline mesh worker failed after the human and depth phases had completed.
Object generation now receives its own HF/Torch cache directories. Regression
coverage checks the routing for every GPU phase. Both required object DINO models
were loaded from their correct local cache on CPU with networking/GPU disabled.
The preceding inference outputs were preserved and hash-verified for a downstream
continuation. GPU mesh generation, full-clip tracking and complete-artifact publication have passed on a verified continuation; a second fresh end-to-end clip has also completed successfully.

The backend now exposes `--check-backend`: this imports the real model CLI with
CUDA hidden and requests help, catching transitive package omissions before
inference. It is a dependency check, not a complete model-weight or GPU test.

**R-007 — preflight must not alter inference state:** dependency checks now avoid
creating depth/mesh output directories. The check-only path is covered for all six
model phases, including hidden CUDA and no inference output. This keeps a failed
or repeated preflight from obstructing a later run.


**R-008 — image agreement does not certify 3D continuity:** a full regenerated
mesh can fit the observed silhouette well while its inferred depth jumps between
adjacent frames. A visible-size hypothesis may also remain below one because a
projected silhouette and visible-surface depth do not measure the full 3D object
diameter. Preserve metric, silhouette, temporal, support and human-alignment
checks separately. A passed metric diagnostic permits testing a hypothesis; it
never approves a resize or a robot reference. The current calibration contract
already records `approved_for_use=false`, requires fresh tracking after a resize,
and documents foreshortening and surface/center-depth limitations. Acceptance
review now exercises those distinctions on full new reconstructions. No new
automatic resize or smoothing default has been introduced from this observation.


**R-009 — visualization must not hide retained human geometry:** a keypoint-only
comparison can make a complete parametric human reconstruction appear incomplete.
The new `human_surface.decode_surface` helper decodes the retained MHR parameters
on CPU into full-resolution, full-timeline surface vertices and model faces. It
records native-to-OpenCV unit/axis conversion and content hashes, and uses bounded
batches with a memory-mapped array. Coordinate/transform rejection tests pass;
a real two-frame model decode produced finite forward-facing surfaces with
18,439 vertices and 36,874 faces per frame. Complete source timelines are now
decoded in downstream comparisons; final video verification is recorded there.
The display does not repair pose or establish grasp/contact readiness.

The surface helper also verifies an already-completed cache without importing
Torch. This lightweight path rejects changed input/model or output bytes; the
focused surface suite now passes three tests. The active downstream inference
snapshot remains immutable and records the earlier helper implementation.


**R-010 — regeneration must compete with a retained strong parent:** fresh raw
outputs can improve over a tiny original yet regress against an already repaired
reference. The downstream experiment fixes candidate hashes before regeneration
and evaluates both geometries on identical source frames/masks. It preserves
historical physics evidence rather than imputing it to a new reconstruction.
Reconstruction now publishes a separate `quality` report of producer image
agreement and temporal warnings, with explicit unassessed dimensions and no
readiness approval. All-frame tracker reliability is retained separately. The
new report is CPU-tested; the active inference snapshot is unchanged, so new
publication-path GPU acceptance remains for a subsequent run. No model default,
mesh size, trajectory or training reward was changed from these observations.


A repeated fresh reconstruction with nearly unchanged mesh dimensions can still
produce worse tracking. Diagnose image agreement, depth drift and temporal jumps
before requesting another mesh resize. Keep unsuccessful calibration hypotheses
and their failed validation evidence, so regeneration does not accidentally
reintroduce a previously rejected change. These findings support the existing
separate gates; they do not establish a new automatic correction rule.

**R-011 — a good resized seed may still float:** foreground-depth registration
can improve silhouette overlap while placing a resting object above its inferred
support. Keep resting clearance separate from nonpenetration. The optional
`support_seed.propose_resting_seed` helper requires an explicitly reviewed resting
frame and tries bounded plane-normal translations, retaining actual-parent image,
depth and support vetoes. It returns the original seed on rejection and never
propagates a resting assumption to the trajectory. Synthetic tests cover bounded
selection, parent preservation, insufficient correction bounds and mandatory
phase evidence. A downstream predicted-data seed passed the checks after a small
translation; full tracking and physics validation remain pending.

**R-012 — standalone tracking must share the pipeline's explicit model runtime:**
a direct vendor CLI invocation can import successfully yet search for checkpoints
relative to source code. The reusable `tracking_runtime.tracking_environment`
checks both score/refiner configurations and checkpoints, supplies the explicit
weights directory, offline flags and bounded render/refiner batch settings. The
normal object-motion backend uses it too. Checkpoint/configuration presence is
not a model-loading test: downstream acceptance separately loaded both actual
checkpoints on CPU and checked finite tensors before requesting a GPU retry.

**R-013 — report paired gate measurements, not just failure labels:** improving
average silhouette fit or smoothing a trajectory can still damage previously
good frames, predicted-depth agreement or reconstructed hand separation. Gate
reports now retain the actual maximum paired hand-gap increase and its threshold,
explicitly distinguishing reconstructed hand points from physical contact. The
acceptance rule is unchanged; historical immutable runs retain their original
reports. A rejected temporal proposal must never replace its tracking parent,
even when it greatly reduces acceleration. Keep size/support hypotheses and
local motion acceptance separate, and mark experimental tracking candidates as
review-only when global checks fail.

A downstream full-clip tracking execution completed using the bounded resting
seed and explicit runtime. It improved average image fit, while local regressions
and discontinuities remained. This validates the launch path, not reference or
actor readiness. No new automatic smoothing, resizing or gate relaxation follows
from that result.

**R-014 — validate prompts against the actual RGB before model execution:** saved
segmentation boxes can be numerically valid yet point at the wrong subject. A
wrong person mask can later fail human-anchored depth correspondence, which is a
symptom rather than permission to lower the correspondence threshold. The video
phase now creates a native-resolution first-frame box/point preview and a hashed
prompt-review receipt before segmentation. The reusable helper rejects invalid
coordinates and out-of-timeline prompts; it explicitly leaves semantics unverified
and identifies later-frame prompts that need their own review. CPU preflight
callers should inspect this preview before launching model phases. A downstream
RGB-only prompt correction is retained separately from its failed attempt; the
active frozen inference snapshot remains unchanged.

**R-015 — few jumps can coexist with an entirely wrong trajectory:** a fresh
multi-sequence run completed and published quality reports, but one near-zero
image-agreement trajectory had fewer discontinuities than a much better tracked
sequence. A smooth or slowly drifting prediction is not accurate motion. Keep
image agreement and continuity as independent findings; repair recommendations
must not rank solely by jump count or average over a good initial segment.
Visible-depth calibration can also become unassessed when occlusion leaves too
little reliable evidence. Preserve that result rather than inferring a resize
from the remaining frames or allowing continuity to override missing evidence.
The existing quality flags expose these cases; no acceptance threshold changed.

**R-016 — rejected calibration is not a resize instruction:** inconsistent held-out
extent/depth evidence can reject a size hypothesis even when its fitting views
look plausible. Calibration now records the exact conflicting frame IDs and
per-frame relative disagreement. Rejection keeps the mesh; the runner and review
text distinguish resolving bad evidence from the separate accepted-hypothesis
resize/retrack path. The numeric limits and the mandatory preflight are unchanged.
Thin, nearly edge-on objects require particular care: a foreground mask can
contain mostly background-like depth while the inferred floor remains stable.
Inspect the native RGB/mask/depth together rather than deleting the held-out
failure or treating its proposed scale as an approved correction.

**R-017 — reject contradictory depth without inventing a surface:** a thin rigid
object turning edge-on can receive background-like predicted depth while a
smaller visible component remains consistent with reviewed views on either side.
The experimental `depth_evidence.qualify_depth` requires explicit bracketing
source evidence and substantial consistent foreground; it marks contradictory
visible object pixels missing, preserving occluders, other pixels and original
inputs. It never fills depths from an existing pose or a desired size, and does
not inspect calibration split labels. Insufficient support preserves the input.

A downstream bounded trial retained the original calibration sample IDs/split
and reduced held-out extent disagreement from 0.6771 to 0.1615 without changing
the 0.20 bound. However, a proposed smaller mesh still failed one of twelve native
seed-view gates after bounded registration refinement. Keep depth qualification,
size evidence and pose acceptance as separate outcomes; the depth result alone
is not a validated reference. No default filter, resize, gate threshold or actor
approval follows from this experimental helper.

**R-018 — RGB proposals need independent geometry checks and state rollback:**
qualified depth does not automatically recover object orientation through an
edge-on view. The reusable experimental `tracking_recovery` process-local adapter
uses the existing RGB refiner, checks each proposal independently against native
image, depth, declared support and motion limits, and tries bounded local/global
recovery before stopping. It restores the selected state before the next frame;
a failed tracker update cannot become the next accepted prior. JSONL evidence
records both actual selected states and rejected attempts. The ordinary backend
and its historical source snapshots remain unchanged. Floor support is not
silently treated as tabletop support. Full GPU acceptance is a separate,
thermally supervised downstream trial.

**R-019 — bind the actual exporter before a guarded run:** a container working
directory can outrank a supplied PYTHONPATH and expose an older module with a
compatible import name but an incompatible callable. `tracking_recovery.load_exporter`
prepends the selected packaged engine paths, verifies the exporter's exact source
file, and binds the full observed-seed call signature before model execution.
Preflight now calls that loader rather than merely importing a same-named module.
An initial downstream startup stopped before tracking on this mismatch; its
immutable attempt and source snapshot remain retained. The corrected trial uses
a separately pinned source snapshot and the existing bounded clock lease.
