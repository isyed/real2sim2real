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
