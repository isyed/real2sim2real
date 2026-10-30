# Explaining partial temporal repairs

A smoother trajectory is useful evidence, but it is not sufficient to approve a
reference. A translation fit can remove a large frame-to-frame jump while moving
the object through a predicted support plane or losing a previously good image
match. Keep those outcomes separate in comparisons.

The reconstruction-repair module provides a pure reporting helper:

```python
from real2sim2real.reconstruction_repair.window_diagnostics import explain

explanation = explain(window_report, locked_frame_ids=[first_frame, last_frame])
```

Pass the report returned by `temporal_pose.run_window` or `refine_window`. The
helper uses that report's original thresholds and measured before/after frames.
It reports:

- **Inherited failures:** present in both the original and candidate.
- **Introduced failures:** created by the candidate, including loss of a formerly
  good image match.
- **Resolved failures:** repaired by the candidate.
- **Locked-anchor blockers:** an original image, depth, support or camera failure
  at a frame the optimizer must preserve exactly.

A locked anchor that already fails an absolute constraint prevents any window
from passing that same strict constraint while the anchor and inputs remain
unchanged. Investigate the source depth, support estimate or anchor pose before
repeating the same optimizer. Do not silently unlock the anchor, change its
measurement or relax a threshold to obtain a pass.

This helper is an explanation, not a new acceptance policy. A rejected proposal
stays rejected. It may still be displayed as a clearly labelled partial benefit,
with its scope, missing checks and failed gates preserved. Metric-size calibration,
full-clip continuity, hand/object interaction and physical execution remain
independent requirements. A short tracking prefix must not be extrapolated into
a complete candidate video or reference.

The helper validates matching frame identities, finite measurements and the scope
of locked frames. Tests cover inherited support violations, introduced image loss,
unavailable depth and mismatched evidence. It performs no inference, optimization,
artifact mutation or network access.

## A bounded smaller-correction retry

When a translation proposal smooths motion but slightly worsens a formerly good
image/depth fit, the existing `localized_pose.backtrack_translation` function can
test smaller fractions of that proposal under the **same** acceptance rules:

```python
from real2sim2real.reconstruction_repair.localized_pose import backtrack_translation

candidate, report = backtrack_translation(
    window_inputs, proposed_camera_poses, original_config,
    fractions=(0.5, 0.25, 0.125, 0.0625),
)
```

Rotations and locked frames stay exact. A passing local result remains subject to
metric, interaction and whole-trajectory checks. Fractions are selected using
these validation measurements; they are not an independent final test. If every
fraction fails, the function returns the original poses and retains trial reports.
An invalid locked anchor identified by `explain` is a reason to skip this retry:
shrinking edits to other frames cannot repair that unchanged anchor.
