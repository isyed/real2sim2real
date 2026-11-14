# Baseline-seeded contact-seating experiments

Retain a successful full sequence and its immutable scene/controller hashes.
Author proposals modify data, never simulator code. The reusable
`controller_actor_training.seating_contract.validate_seating_proposal` accepts
only a frozen graph's configuration inputs:

- Explicit object-frame pad displacement vectors, each at most 3 mm in norm;
  the existing fitter must enforce final joint bounds relative to the baseline.
- Alternatively, direct closure offsets on explicitly allowed joints, at most
  0.12 rad each and 0.24 rad in total vector norm. Do not combine modes.

These are initial experimental search bounds, not a claim of safe physical
motion. Preserve native joint limits as well. No revision may accumulate offsets
beyond the original baseline bounds. Graph tools, timing, scene, pressure,
release logic and acceptance limits remain unchanged. Missing or nonfinite
parameters reject the proposal.

Zero edits must bypass fitting and reproduce the original command/state stream.
A controller replay check and an actual graph execution check are separate;
neither substitutes for the other. Keep one persistent simulator session through
the full sequence. Never insert resets or settling pauses at graph boundaries.

Fit, check the complete obstacle-restored closure/path, and then run loaded
physics. Excellent static pad alignment can still fail during inversion.
Preserve rejected trials and the successful comparison. Before advancing to
control acceptance, audit contact quality, geometry uncertainty and robustness
separately from nominal task completion.

A bounded feedback campaign can use one initial baseline-seeded proposal plus
four revisions. Each round receives measured simulation diagnostics and the
previous proposal. Invalid proposals consume their round without running
physics. Reference scans, ground-truth trajectories and evaluation scores stay
outside the author packet and proposal selection. Score frozen outputs separately.

Private downstream projects own the exact author packet, execution adapter,
permissions, robot geometry and campaign budget. Preparing a packet does not
send it to an external model or validate an unimplemented adapter.

## Zero-edit graph execution

`seating_graph_replay.FrozenSeatingSession` supplies the configuration and full
sequence tools for the existing GaP v3 seating seed. `execute_zero_seating_graph`
loads the optional installed GaP scheduler with one worker and checkpoints off.
It validates the complete seed before execution and rejects nonzero inputs.
Configuration does not step physics or run fitting. The sequence callback runs
exactly once, owning every internal phase and its continuous simulator clock.
This coarse graph contains one full-sequence node; it does not test hypothetical
per-phase graph handoffs. The downstream callback owns finite execution deadlines.

The graph status and the callback's physical task verdict remain separate. A
graph can finish while a physical task fails. Verification must compare fresh
commands, state arrays and timestamps against immutable baseline outputs, check
release/home outcomes, and retain source hashes and native scheduler traces.
No fitting, saved-state playback or output copying substitutes for fresh physics.

Passing this check establishes the zero-edit execution path only. Nonzero
proposal fitting, full path checks and candidate execution remain separate
adapter responsibilities. Tests in `test_seating_graph_replay.py` cover ordering,
duplicate execution, nonzero rejection, and actual-scheduler physical-failure
propagation when the optional GaP dependency is installed.
