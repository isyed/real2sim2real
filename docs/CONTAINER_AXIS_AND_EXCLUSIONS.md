# Container endpoints and provisional protrusion exclusions

`real2sim2real.egocentric.container_axis` complements the tapered-container
hypothesis without changing its dimensions. `fit_endpoint_pose` takes image
coordinates ordered **bottom, rim**, positive per-endpoint pixel uncertainties,
calibrated intrinsics, fit-only depth points, and a starting rigid pose. It tests
both the original and reversed-axis initialization. Robust endpoint costs are
soft constraints; the exported endpoint errors must still be inspected. They
are fitting residuals, not an independent validation score.

Retain annotation origin and visibility. An endpoint center inferred from an
occluded ellipse is less certain than a directly visible bottom. Do not label
assistant-selected pixels as an independent human evaluation set. Keep depth and
image acceptance limits unchanged and retain the previous hypothesis.

`axis_transition` compares directed axes in a **shared fixed world**, using
measured capture-time differences. Compose camera-relative fits with accepted
background camera poses first. Symmetry-axis yaw is excluded intentionally.
Sparse interval averages do not certify a continuous trajectory or bound peak
rates. Do not smooth across a real inversion merely to eliminate an axis change.

`exclusion_bands` converts finite protrusion samples in object coordinates into
provisional axial/radial keep-out bands with explicit margins. All azimuths are
excluded when yaw is unresolved. Margins are configured assumptions, not measured
confidence bounds. Empty observations remain unavailable, not clear. Samples may
include neighboring body pixels and rejected pose errors; the result explicitly
does not verify collision geometry or exclude every hidden protrusion.

Promote neither rejected poses nor their transformed protrusion bands into a
control-ready scene. Before grasp planning, validate the shape and poses, check
the relevant protrusions from useful views, and recheck finite pad support and
insertion clearance. Tests cover recovery from inverted initialization,
yaw-invariant directed-axis diagnostics, and unknown-geometry handling.
