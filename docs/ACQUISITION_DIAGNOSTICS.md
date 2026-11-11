# Compare acquisition before changing loaded-arm control

`acquisition_compare` compares two pinned contact-audit replays without running
new dynamics or changing acceptance. One can be a failed candidate. Each audit
must reproduce its source state archive byte for byte; original/replay result
hashes and both archives are verified before interpretation. A complete native
clock and the same diagnostic loading thresholds are required.

```bash
python -m real2sim2real.controller_actor_training.acquisition_compare \
  --config /data/acquisition-comparison.json
```

The downstream configuration gives `baseline_audit`, `baseline_sha256`,
`candidate_audit`, `candidate_sha256`, `allowed_configuration_changes`, `output`
and `deadline_s` (at most 120 seconds). For a posture-transition comparison, an
allowed change can be `contact_force_feedback.posture_transition`. Configuration
paths are explicit, not wildcard exceptions. Planning-output locations and the
read-only contact-logging option are excluded from control-setting comparison.

The pair must preserve compiled scene identity, payload, sliding friction,
timestep, requested duration, native sample times, phases and pad identities.
Both planned wrist frames, open joint coordinates and object targets stay fixed
to numerical tolerance. Closed finger postures can differ intentionally. Changed
reference/controller settings must be explicitly declared; the stage does not
reinterpret them as a matched controller. This is an observational comparison,
not a source of new admission evidence.

The report separates approach, closing, lift and hold. For each named pad it
records loaded-sample fraction, first loaded sample, coherent mean/peak force,
active pressure deficit, pressure-offset norms, sampled material sliding and
object-local contact-centroid migration. Other hand/object contact meshes remain
visible. Sliding integrates only consecutive loaded samples within the declared
gap and the same phase; unloaded gaps are not interpreted as zero slip.

Paired samples report pressure-offset differences, contact-centroid separation
where both pads are loaded, and actual object-pose differences in the measured
wrist frames. These are invariant to a common world-frame transform. Diagnostic
pose changes from first-close/first-hold samples do not replace the source drift
anchors or acceptance maxima. A commanded posture endpoint, an offset vector and
an actual material contact are distinct observations.

Force timing is explicit: contact-patch forces use coherent post-step
`mj_forward` samples. The latest servo request, offset and controller force
observation precede that state by one physics step. Approximate pressure deficits
are reported only after control activation; they are not exact simultaneous servo
errors. Contact events have the recorder's sample resolution. Low sliding during
poor pad coverage does not prove a better grasp.

`review.html` is self-contained and offers pad/window selection, a native-time
scrubber, pressure/contact plots, contact centroids and per-phase summaries.
Inactive pressure targets are masked in plots. No CDN or browser GPU is required.
The HTML template is included in the module package. Outputs preserve source
verdicts and declare that diagnostics cannot promote them.

Use acquisition evidence to select a bounded controller experiment: for example,
maintaining bearing-pad contact while posture changes, rather than simply raising
all pressure goals. The comparison alone does not isolate causality or justify
loaded-arm trials. A revised controller still needs stationary, nominal motion
and frozen-matrix evaluation under its original acceptance criteria.
