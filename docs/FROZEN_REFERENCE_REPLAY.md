# Standalone replay of a retained manual controller

`controller_actor_training.sampled_reference.SampledReferenceReplay` executes a
portable, already-planned reference with fresh contact feedback. The
`agentic-policy` backend `frozen_reference` exposes one graph operation,
`controller.run_reference`, for comparison-only replay. It executes once,
continuously, with no graph boundaries between task phases.

This preserves the retained controller's clock, interpolation, pressure updates,
support-triggered release, placement-gated home return and diagnostic sampling
order. It is separate from the broad `continuous` backend used by from-scratch
proposals. Replay success establishes preservation of this controller, not a
new policy solution or strict quality/control acceptance.

## Portable inputs

The trusted experiment selects relative `scene`, `initial` and `reference`
paths within its bundle. It supplies robot/body/pad names, pressure groups and
bounds, release settings, phase thresholds, task/quality checks, home limits and
a finite deadline. Geometry, gains and criteria cannot be edited through graph
commands. The input paths must already have screening provenance: replay is not
a fitting or collision-planning operation. The object remains free, without
weld, mocap or external assistance. Scene and declared timestep must match.

The initial-state NPZ contains exactly one `qpos` and `qvel` vector. The reference
NPZ contains only these planned-input arrays:

| Array | Meaning |
| --- | --- |
| `feedback_times_s` | Exact clock from successive physics-timestep additions |
| `reference_times_s`, `previous_reference_times_s` | Explicit retiming at current and previous feedback ticks |
| `finger_targets_rad` | Nominal posture before pressure/release feedback |
| `object_targets_world` | Desired 4x4 object poses used for tracking diagnostics |
| `phases` | Nominal phase labels |
| `arm_times_s`, `arm_q_rad` | Planned arm interpolation knots |
| `pre_times_s`, `pre_q_rad`, `post_times_s`, `post_q_rad` | Screened home approach/return paths |

Joint ordering, finite values, dimensions and the exact clock are checked.
Future rollout states or recorded actuator-command arrays are rejected. Contact
forces, pressure correction, release timing, withdrawal timing and home gates
are recomputed from evolving physics. References are frozen; feedback remains
active. Private legacy scripts may construct the input bundle once, but those
scripts and neighboring checkouts are not runtime dependencies.

Keep the comparison bundle and its successful manual grasp/path parameters
outside the from-scratch author's inputs and tools.

## Validation

`agentic-policy/tools/manual_replay_acceptance.py` runs the graph from installed
wheels with workspace checkouts hidden. Expected states must remain outside the
simulation sandbox. It records unchanged input hashes, graph/tool execution,
physical task/home results and full physics output.

Afterward, `controller_actor_training.replay_comparison.compare_replays` checks
finite qpos, qvel, commands and timestamps exactly, including dtype/dimensions.
It separately checks task, release, home, warning/failure history and every
diagnostic record. Explicit aliases can reconcile legacy field names such as
`rack_upward_force_n` with `support_upward_force_n`; values never change.

Audit compiled-model arrays after relocating assets: path offsets and associated
buffer bookkeeping may differ, but geometry/physics must match. Preserve source,
input, wheel and output hashes. Conditional geometry and strict-quality failures
remain visible when nominal completion and replay pass. This establishes no
new robustness rate, training admission or physical-control authorization.

For an explicitly manual-seeded improvement experiment, use the separate
[reference refinement backend](REFERENCE_REFINEMENT.md). The comparison-only
backend remains unchanged; an empty refinement must reproduce it before any
author-generated pressure or finger-posture profile is evaluated.
