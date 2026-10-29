# Reproducibility and adoption

Record the Git commit, installed package versions, model hashes, robot asset
hashes, driver/CUDA versions, input video hash, prompts, configuration and
random seeds for each real run. Hash receipts detect changed files; they do
not guarantee bit-identical GPU physics across hardware or library versions.

The extraction source inventory is in `provenance/extraction.json`. It records
original file hashes and final adapted hashes without paths to private source
workspaces. `provenance/release-files.json` covers the finished implementation.
Repair algorithm version 0.10.0 is retained inside the new distribution's
0.1.0 API release.

Use versioned installation of this repository in downstream applications.
Keep their dataset adaptation, campaign configuration, job control and score
reporting outside this project. Existing running jobs and historical outputs
should remain pinned to their original code. Move a downstream workflow to
this implementation only after a matched acceptance test; do not rewrite
past artifact provenance to pretend it used the new package.

Recommended first acceptance run: one short RGB clip through reconstruction,
metric evidence review, repair and retargeting, followed by frozen-controller
physics before allocating actor training. Use separate output directories for
raw and repaired alternatives. Check image agreement, metric size, support,
continuity, contact and unassisted physical behavior independently. A learned
actor may compensate for modest tracking error, but cannot validate incorrect
geometry, coordinate transforms or a missing support surface.

Retain the actor, resolved configuration, parent robot reference and external
model/robot asset identities for inference. Replay is optional for future
training: a fresh buffer can be regenerated, but that is a different
continuation from restoring the exact learner/replay state.
