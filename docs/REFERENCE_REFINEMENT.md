# Seeded reference refinement

The `agentic-policy` backend `reference_refinement` lets a reviewed author refine
an existing successful `SampledReferenceReplay` controller. It is separate from
`frozen_reference` comparison-only replay and from the broad `continuous`
from-scratch interface. This is controller refinement, not residual RL.

`controller.run_reference(edits={})` executes one uninterrupted reference sequence.
Empty edits leave the original physics/control path unchanged. A nonempty edit
contains increasing `reference_times_s` knots covering the retained controller's
pressure-start through release reference times. Optional profile values are:

- `pressure_targets_n`: one target per configured pad, always in (0, 5] N.
- `gain_scale`: scalar normal-pressure response multiplier at each knot.
- `finger_offsets_rad`: per-finger command offsets; only joints in the trusted
  pressure groups may change.

Values interpolate linearly. Endpoint corrections are zero, endpoint pressure
and response return to baseline, and outside the interval the retained commands
are restored. Actual supported-release detection, pressure memory, home return,
arm paths, physics and evaluation remain owned by the host. Pressure feedback
still responds to measured contact forces; the supplied profiles themselves are
time-dependent, not a learned observation-conditioned policy.

The trusted backend configuration requires `refinement_bounds` specifying
`maximum_knots`, `minimum_gain_scale`, `maximum_gain_scale`,
`maximum_finger_offset_rad` and `maximum_offset_speed_rad_s`. The author cannot
change those bounds through the tool. Validation rejects unknown fields,
nonfinite data, incorrect dimensions, unowned joints, native joint-limit
violations and excessive correction rates evaluated on the actual feedback
clock. Gain-scaled admittance must remain below the existing 0.01 m/N/s ceiling.
Invalid proposals cannot step physics. Their report saves only the actual
initial state and explicitly records that no execution occurred.

This interface deliberately preserves the arm path and timing. If development
failures require wrist replanning, pose adaptation, or new grasp topology, use a
separately declared broader adapter rather than claiming this profile interface
can solve them. Keep nominal success as a mandatory retention gate; compare the
same development cases and strict criteria before promotion. Do not turn a
failed perturbation into success by changing thresholds.

For human-in-the-loop authoring, enable `human_review_between_proposals` and
require a review receipt tied to the prior result hash. Host-run baseline
verification does not consume an author proposal. An author submission containing
empty edits does consume its proposal slot, but is a fallback rather than a new
controller improvement. Two proposed refinements mean two author submissions,
not a new from-scratch candidate plus two revisions. Keep held-out evaluation outside the
author packet, including when the baseline's private planned references are
explicitly approved for use as a seed.

Report distinct edit profiles separately from author-call counts. Repeated empty
edits can confirm repeatability but cannot establish a new robustness gain. When
the author repeatedly retains the baseline, review the available evidence and
interface before adding more proposal budget; this is not proof that no better
controller exists. New campaigns can enable `author_loop.stop_on_duplicate_edits` and declare
verified `baseline_edits` to stop repeated effective profiles before physics.
See [author-loop controls](AGENTIC_AUTHOR_LOOP.md); legacy campaigns without
this opt-in retain their original behavior.

Returning a target or gain to its original value does not restore the preceding
contact state or pressure memory. Even a brief early correction can change a
later grasp outcome. Conversely, a correction starting after a failed trial ends
cannot repair that failure. Compare saved commands and states before intervention
and retain the complete task outcome as the first gate, before interpreting
small improvements in individual metrics.

For bounded observation-conditioned posture rules, use the separately declared
[contact reference adapter](CONTACT_REFERENCE_REFINEMENT.md). Verify exact zero
correction before switching a matched comparison to that backend.
