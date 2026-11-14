# Stream-offset audits with static and moving controls

`real2sim2real.egocentric.temporal_alignment` supports read-only alignment audits
with fixed geometry and rig calibration. Declare offsets and their sign before
scoring: comparing source frame `t` to target frame `t+d` uses positive `d` for a
later target frame. Record actual supplied capture-time differences separately
from nominal frame duration; exposure timing may still be unknown.

`track_points` makes direct forward/backward optical-flow tracks from one anchor
image to every offset. `common_cohort` retains the same identities across the
whole sweep; failed offsets cannot silently change a score's denominator.
`track_patch_series` selects local anchor corners once and uses their common
cohort to estimate patch-center motion. It requires at least three corners and
median motion scatter ≤2 px. It estimates local appearance motion, not an
independently verified physical center. Keep direct and patch-based attempts
separate and do not choose a tracker by its preferred offset.

Static scene controls are essential when the rig moves. `static_registration`
fits a diagnostic source-3D-to-target-image pose on one subset of static
landmarks and reports held-out errors on the rest. It needs at least 12 fit and
8 held observations and robust PnP support. This pose can absorb extrinsic error
as well as head motion. It is **not a calibration update**. Report both raw
calibrated projections and statically registered projections for moving markers.

Declare static image regions independently of reprojection errors, avoid moving
hands/objects, retain landmark identities and report common sample counts.
Source marker depth estimated from neighboring stereo points is uncertain and
may mix the marker and body. Complete marker identity and physical 3D location
must not be inferred merely from a good optical-flow cycle.

Report full offset curves, per-anchor minima and missing data. A flat curve from
a nearly stationary object contributes little timing information. Different
per-anchor optima, or small improvements relative to annotation uncertainty,
do not justify a global time shift. Data used to choose a tracker or offset are
development evidence; keep a separate diagnostic anchor when possible.

Tests exercise known image shifts, fixed-cohort handling, held-out static pose
recovery and patch-corner tracking. Geometry, rig calibration and accepted
reconstruction artifacts must remain unchanged during the audit.
