# Palm tracking and contact retention

`controller_actor_training.grasp_diagnostics` separates two measurements:

- Palm error relative to the moving object's intended reference anchor, including
  orientation error. This distinguishes object drift from a world-fixed target.
- Recorded contact intervals, including the longest uninterrupted contact during
  the intended contact phase. Many short impacts are not one sustained grasp.

Inputs are aligned recordings in metres with xyzw quaternions and an explicit
recording rate. The caller must use the same control clock for states and phases.
The evaluator saves physical states after each control step and kinematic states
at the corresponding reference index; interpret fast events with that one-step
sampling distinction in mind. Palm link origins must match the controller's
semantic palm frame; apply body-to-frame offsets first when they are nonzero.

The target anchor is not automatically a verified object handle. Inspect it
against reconstructed geometry and source observations before moving a wrist
target. Contact events do not prove force closure, lifting or task success.
Check body drift separately: stronger fingers or bounded arm corrections cannot
compensate for a robot that has moved far away from the object.

For intervention selection, first diagnose the retained controller, preserve its
scene and reference, and change one mechanism at a time. Repeated successful
contact plus retained balance and task scores is stronger evidence than a lower
static palm error alone. Defaults and saved policy artifacts remain unchanged.

## Diagnose the approach before changing the grasp

Measure physical and reference pelvis displacement over the initial approach,
alongside upright, intended-phase palm error from `reach_audit.summarize`.
Restrict phase/contact summaries with the same explicit upright-frame mask.
An apparently correct kinematic approach can coexist with very little physical
translation; stronger closing cannot resolve that separation.

The current raw SONIC tokenizer supplies future joint positions/velocities and
relative root orientations, without directly encoding absolute world root XY.
It does not guarantee reproduction of the reference's world-space translation.
Inspect the source gait and retargeted legs as well as controller performance.
Any opt-in joint-reference blend is a separate physical experiment: preserve
scene/clock/hand settings, bound the correction and require repeated gains in
reach, balance and task tracking. Do not enable it as a generic fix merely
because the reference robot walks in a kinematic video.

Compare stance evidence with world-space foot-landmark velocities. Persistent
motion of both near-ground feet can indicate an inconsistent walking reference,
but low height is not measured contact and must not be converted into a planted
foot label automatically. A fresh IK export that reproduces the old joint
trajectory is a reproducibility check, not evidence that the gait was repaired.
When testing joint-target corrections, reject an apparent palm-error gain that
comes from earlier falling or worse task tracking.

Reconstruction admission also needs a complete source timeline. A repaired prefix
is useful diagnostic evidence, but a missing or rejected tail must remain
explicitly unavailable rather than become a carried pose or training target.

## Continuous contact and intended phase

`longest_contact_s` measures uninterrupted contact *within* the intended contact
phase; a gap in the reference phase ends that interval even if physical contact
continues. `longest_physical_contact_s` and `physical_contact_intervals` measure
the actual uninterrupted contact timeline regardless of phase. Use both: phase
fragmentation must not be mistaken for a physical loss of grasp.

