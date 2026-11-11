# Bearing support geometry before control

`bearing_support_plan` explores a different support contact family without
changing an admitted controller. A lower surface can carry upward load through
its normal, whereas a vertical wall relies on friction. Neither mechanism alone
establishes a stable robot grasp.

The stage consumes a hash-pinned fixture grasp result and its verified prepared
scene, settled object pose and source plan. It preserves the source primary hand
at its closed pose while fitting a new support hand. This is an initial geometry
screen, not simultaneous acquisition, arm reachability or loaded control.

Each proposal declares five contact XY coordinates in the upright metric world,
a horizontal yaw and `orientation`: `palm_up` or `vertical_wrap`. The latter keeps
the palm beside the object with the long hand axis downward. Pad frames come
from the source hand's actual collision hulls; contact positions must respect the
embodiment's finger order. The thumb targets an exterior upper surface and the
four other pads target lower surfaces.

First ray intersections from outside reject hidden cavity floors. Five rays per
contact check a 4 mm stencil, exterior normal direction and height continuity.
An optional bounded grid surveys lower surfaces with at least 15 mm initial gap
above the declared tabletop. These are reconstructed geometric hypotheses,
not independently observed patches or continuous surface certificates.

The planner fits open, seated and slightly preloaded configurations using native
joint limits. It checks an 80 mm insertion and seating path at no more than 2 mm
translation / 0.02 rad joint increments. The original source position and
penetration limits remain in force, along with 25 degree pad-normal alignment
and finite collision-pad contact samples on every pad. `path_audit` now records
the worst colliding geometry pair, segment, interpolation fraction and full owned
joint coordinates; it retains its original sampling and maximum calculation.

Static gravity screens combine the unchanged primary contacts and hypothetical
support contacts at nominal friction and zero friction. Explicit normal ceilings
are planning assumptions, not requested pressure or motor feasibility. The wrench
helper reports each point's world force and force/torque equilibrium residual.
Per-pad sample groups share a force budget; a patch sample is not a new actuator.
An LP solution is one feasible static allocation, not the realized controller
load sharing or a proof of force closure.

Run locally with a CPU-capable MuJoCo environment:

```sh
python -m real2sim2real.controller_actor_training.bearing_support_plan --config /private/config.json
python -m real2sim2real.controller_actor_training.bearing_support_review --plan /private/output/plan.json
```

Required config: `source_trial: {path, sha256}`, `proposals` (1–12, each
`xy_m` with five two-dimensional coordinates and `yaw_rad`),
`maximum_evaluations` (integer, 1–200), `maximum_normal_force_n` (0–10 N),
`deadline_s` (0–1800 s), and fresh `output`. Optional `survey_xy_m` has at most
4096 points. Positive upper-bounded quantities exclude zero. Outputs include all
rejections, fitted waypoints, finite samples, static forces, source configuration,
code snapshots/hashes and runtime versions. Source implementations must remain
unchanged during execution. Outputs never overwrite an existing directory.

Every result explicitly sets `control_admitted: false`. A promising geometric
proposal still needs independent contact evidence, full coupled acquisition and
arm-path checks, actuator feasibility, then complete unchanged stationary
acceptance. Only a matching stationary pass permits the moving-reference retry.
Release from a different contact strategy cannot be inherited.

## Distinguish a stationary review from a stopped video

`arm_replay_review --refresh-page` reuses the saved video and validated playback
cache. It refreshes the report text without rendering or integrating physics.
Prior review result/video hashes must still match. Stationary reports state the
first hold time and saved final time prominently: commanded motion deliberately
stops while the free object's physics continues. A moving replay is labeled
separately and does not imply task success. This prevents stationary retention
videos from being confused with truncated demonstrated-motion replay.
