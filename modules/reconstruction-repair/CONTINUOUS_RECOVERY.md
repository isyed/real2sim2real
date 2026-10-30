# Bounded continuous local tracking recovery

`local_registration.propose` refines a discrete failed recovery hypothesis using
a CPU pose search. It fits translation and rotation only; it never resizes the
mesh, changes camera/depth, moves the support plane or edits human motion.
The previous pose must be a source-derived accepted state. Source identity and
seed provenance remain the caller's responsibility.

Defaults allow a 45-second optimization search and at most 180 objective calls.
The objective uses a reduced **complete** image grid and the current source's
observed foreground depth. It does not crop away extra projected geometry.
After search, it rasterizes native triangles and recomputes native silhouette,
depth, support, camera and the existing 8 cm / 30° motion checks. The final native
evaluation can extend elapsed time beyond the search deadline. The caller still
owns a hard process/container deadline.

A CPU proposal is not a complete repaired trajectory or metric-size approval.
It cannot certify itself for physical execution. The optional integration is:

```text
python -m real2sim2real.reconstruction.tracking_recovery \
  --export <prediction-only-export> --depth <own-depth.h5> \
  --seed <verified-initial-pose.json> --support <predicted-support.json> \
  --output <new-poses.pkl> --continuous-cpu-recovery
```

This explicitly composes the reconstruction and reconstruction-repair modules;
both module packages must be available in the worker's Python environment. The
default tracker does not import or run the optional optimizer.

When discrete local recovery fails, the integration may make at most four CPU
recovery calls per sequence. It selects an already motion-bounded local seed,
then independently checks the returned pose through the **actual GPU renderer**
and the same float32 image/depth/support/motion gates used for other candidates.
It never trusts `proposal_valid_for_tracking` from the CPU report. A GPU rejection
falls through to existing bounded global registration; total failure still stops
tracking. No unvalidated pose is carried forward. Per-call evidence, GPU checks,
budgets and metadata are retained in recovery records.

This optional search can address a coarse discrete proposal grid. It does not
solve complete occlusion, inaccurate masks, wrong scale, invalid depth, impossible
motion or a missing support. Preserve their warnings. Fitting a known failed
frame is a development diagnostic, not an unseen held-out evaluation; downstream
clips and independent checks are required to establish general benefit.

Tests cover numeric pose improvement with unchanged limits, failure on wrong mesh
size, out-of-bound seeds, provenance/grid checks, and acceptance versus rejection
of a CPU proposal by the independent GPU-adapter boundary (CPU stand-in tests).
Actual GPU execution remains a separate guarded integration check.
