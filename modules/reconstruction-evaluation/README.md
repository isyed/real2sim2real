# Reconstruction evaluation

CPU-only offline scoring, separately installed from inference. The initial
implementation scores one rigid object against a nested-object Parquet reference
and a reference scan. Camera motion and object motion are composed before scoring.
Reference assets must never be mounted into inference containers.

`r2s2r-evaluate --run RUN --reference-motion MOTION --reference-mesh MESH
--object-name NAME --output OUTPUT` reports geometry Chamfer, trajectory ADD/AUC,
mean position and orientation errors, MPPE, and SP/MP success thresholds. Meshes
are registered with deterministic surface sampling and multistart trimmed ICP.
This local implementation does not claim bitwise parity with a benchmark's
registration code. Mesh similarity registration never rescales rollout motion.
Exactly one initial rigid transform aligns worlds; no per-frame alignment occurs.

Only ground-truth-visible frames are eligible. Missing predictions reduce explicit
coverage and failure-penalized AUC; incomplete sequences cannot pass success rates.
Continuous errors over observed predictions are reported separately. Single-object
RPE is zero by definition and is not informative. Multi-object symmetry-aware
benchmark evaluation is a future extension, not silently approximated here.

Source frame IDs, timestamps, reference units and quaternion order are explicit.
The nested reader defaults to metre poses with wxyz quaternions; other datasets
can supply their column names and convention. Scores preserve source hashes and
per-frame errors. Evaluating a reconstruction is not evidence of policy execution.

AUC uses the nine strict ADD thresholds from 1 through 9 cm, with trapezoidal
integration. Registration, sampling and reference anchors are local approximations;
these scores must not be presented as official leaderboard results.
The evaluator refuses to overwrite an existing score directory and records both
input and implementation hashes. `--camera-mode fixed` provides a diagnostic
ablation using the same object poses without estimated camera motion.

Install the `plots` extra and run:

```bash
python -m real2sim2real.evaluation.report \
  --run /data/predictions/run --scores /data/scores/run --output /data/reviews/run
```

The report contains exportable SVG/PNG error plots and an HTML frame slider for
rendered object overlays. Keep the prediction directory alongside the report;
overlay links are relative. Use `--comparison /data/scores/fixed-camera` to plot
the diagnostic camera ablation as a second trace.
Set `--label` and `--comparison-label` to identify other tracking ablations.
Plots retain the full declared time axis when a failed reconstruction has missing
predictions, rather than shortening the displayed episode to its accepted prefix.
