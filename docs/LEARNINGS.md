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
