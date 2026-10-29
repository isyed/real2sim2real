# Module learning register

Record reusable findings here. Scenario identities, private videos, measured
results and deployment settings belong in the downstream experiment repository.
An implementation change is not evidence of improved reconstruction or robot
performance until the corresponding acceptance test has run.

| ID | Finding | Module change | Evidence and state |
| --- | --- | --- | --- |
| R-001 | Independently installed perception models may require incompatible environments. Selecting subprocess interpreters only for the GPU calls still leaves heavy export/packing imports in the parent environment. | Reconstruction now dispatches every phase, including packing, metric input preparation, object export and artifact publication, to an explicit worker environment. `backend_commands` accepts argument arrays for externally supervised container wrappers; `backend_pythons` remains supported. | CPU dispatcher tests and offline imports in separate segmentation, mesh and tracking environments pass. Full model execution through this dispatcher is pending. |
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
