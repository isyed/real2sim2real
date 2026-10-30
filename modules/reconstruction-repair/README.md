# Reconstruction repair

The complete portable repair engine, algorithm version 0.10.0, is included.
It has no dependency on a neighboring workspace or cloud storage. The pipeline
adapter consumes the common reconstruction artifact; the low-level CLI also
accepts the engine's native bundles:

```bash
python -m real2sim2real.reconstruction_repair.cli --help
```

The normal gate order is metric-size calibration first, bounded fine object
pose/continuity repair next, and optional human repair last. Size evidence is
retained separately from later tracking decisions. A proposed resize requires
retracking and revalidation; a better silhouette alone does not certify depth.

Each run records accepted, rejected, disabled and unavailable stages, reasons,
per-frame diagnostic values and the accepted output bundle. Parent hypotheses
are preserved. The engine includes silhouette-preserving temporal correction,
RGB pose hypotheses, rotation/translation continuity checks, support placement
diagnostics, and native-MHR human fitting with explicit evidence. Not every
capability is enabled for every input.

See [the evidence contract](../../docs/DATA_CONTRACTS.md). The orchestrator
requires an explicit `metric_evidence` NPZ and reviewed `floor_roi_source`;
it does not invent a floor plane to force a pass. Human fitting additionally
needs a hashed `mhr_decoder` specification, appropriate human observations,
`human_metadata` (native-to-world transform, decoder identity and parity
provenance), and independently exported `human_parity` geometry. These are
checked before fitting; the adapter cannot manufacture its own evidence of
decoder correctness.
Its decoder is CPU-only. Set `execution.cpu_only: true`.

The accepted output is a reconstruction candidate, not an actor-training
approval. Reference contact, physical contact, object stability and task
completion must be evaluated separately.

For an independently calibrated mesh whose fresh seed floats above a reviewed
resting surface, `support_seed.propose_resting_seed` offers a bounded CPU
plane-normal search. The caller must explicitly identify the frame's resting
phase; the helper does not infer or propagate contact. It selects the smallest
sampled translation passing the actual parent's image, depth and support checks,
returns the original seed unchanged on rejection, and records every attempted
move. Default maximum translation is 5 cm. This is a seed hypothesis requiring
fresh tracking, not a trajectory repair or permission to train a controller.

After selection, `real2sim2real.comparison_video.render` can show source RGB
beside complete before/after object and human surfaces. Decode saved humans with
`human_surface.decode_surface` first. The CPU renderer preserves all source
frames, checks matching timelines, verifies the encoded video by decoding it,
and caches only matching video/geometry/surface identities. Its transparent
silhouettes do not establish occlusion, grasp or physical correctness.

A **rejected calibration** is distinct from a validated resize hypothesis. Its
report records `inconsistent_extent_frame_ids` and per-frame relative scale
disagreement. Keep the input mesh and inspect those native RGB/depth/mask views;
do not apply the displayed scale multiplier merely because it was calculated.
Normal fine repair remains blocked until current-size evidence is consistent.
The runner recommends resize/retracking only when an independently retained
calibration hypothesis passes its evidence checks and requires a different size.
