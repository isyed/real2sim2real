# Foreground-depth and visibility preflight

Run a CPU audit before treating a complete object trajectory as observable:

```text
python -m real2sim2real.reconstruction_repair.foreground_visibility \
  --input <portable-reconstruction> --output <new-audit-directory>
```

The artifact boundary streams the complete original timeline, predicted native
object/person masks and its own PNG-encoded millimetre depth from the declared H5
camera stream. It records input hashes and does not run a predictor or controller.
Numeric APIs `measure` and `audit` can also be used with other explicit metre-depth
readers; no dataset or cloud account is hardcoded. Optional H5/Pillow imports are
loaded only for the artifact reader. Existing outputs are not overwritten.

The audit records exposed foreground support, depth median/spread, missing-mask
intervals and temporal depth warnings. Defaults retain the existing two-pixel
erosion, 100-depth-pixel minimum and 0.35 relative IQR threshold. A separate warning
compares adjacent surface-depth medians against 2.4 m/s times elapsed time plus
0.02 m sampling slack. This is a **warning**, not a measured object-centre speed:
different visible surfaces can legitimately have different depths. It must not
be used to prove that true motion is impossible.

An available observation is not proof of metric accuracy. A uniformly predicted
background patch may have thousands of samples and a narrow depth spread.
`full_observation_workflow_ready=False` identifies an unsuitable input for a
workflow requiring visible, coherent evidence at every frame. It does not veto
all possible reconstruction methods. Missing masks can mean real occlusion or
segmentation failure; inspect the source RGB before choosing a remedy.

The report names gaps and where a fresh registration would be needed after each
gap. Its `duration_s` is the timestamp span from the first missing sample through
the last, not an interpolated trajectory or an authorization to bridge it. The
original frame IDs and inclusive frame counts remain explicit. No hidden pose,
new depth, mesh scaling or reference approval is generated. `overlay` visualizes
observed mask pixels without drawing an object where none was observed.

## Qualification and the next stage

The existing `depth_evidence.qualify_depth` may mark contradictory values missing
only when independently reviewed bracketing depth anchors and a substantial
consistent foreground component exist. It never fills a surface with interpolated
depth. If the consistent foreground is insufficient, leave the original input
and retain that blocking result. Do not progressively widen the criteria until
a desired trajectory passes.

A bounded integration experiment found an interval where qualification could
not repair the failed frame. A later fresh translation seed independently passed
native CPU image/depth/support checks, but observations were still absent in the
middle. That is a segmented-reacquisition candidate, not a full repaired track.
Hidden motion needs a separately specified and validated inference model, such
as independently checked human/object interaction constraints. A pose copied
through a gap or validated against depth synthesized from that same pose is not
an independent measurement.

The common sequence is: metric-size evidence, foreground visibility/depth audit,
bounded qualification if evidence supports it, independent registrations for
observable segments, and explicit inference/validation for any hidden interval.
Only then assess the complete paired geometry/reference for downstream physics
or controller training. None of these audits automatically authorizes a new
selected reference.
