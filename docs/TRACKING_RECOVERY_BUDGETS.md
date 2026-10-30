# Finite tracking recovery allowances

A long clip can require many locally recovered updates even when every accepted
pose satisfies the original image, depth, support and motion checks. The default
180-frame recovery allowance is a computational limit, not a geometric limit.

The experimental reconstruction tracking-recovery CLI accepts
`--recovery-frame-budget N`. It validates that N is positive and no greater than
the source update count (source frames minus one). Omitting it preserves the
180-frame default. The selected allowance is written into both success metadata
and failure records, so callers can check the actual protocol rather than infer
it from clip length.

This option changes only the finite number of recoverable updates. It preserves
all image/depth/support/step thresholds, the 12-registration cap, the opt-in CPU
proposal limit and independent GPU revalidation. The caller must still enforce a
wall-clock deadline and thermal guard. Increasing the allowance is not evidence
of accuracy, full-clip coverage, valid metric calibration or physical execution.
A failed or partial run must retain its original reference and missing-pose scope.

The lightweight `configured_limits` API validates the allowance; the tracking
entry point additionally validates it against the actual input frame count.
Tests verify unchanged geometric limits and reject invalid or oversized budgets.
