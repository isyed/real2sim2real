# Contact-conditioned reference refinement

The opt-in `contact_reference_refinement` backend keeps the sampled reference,
arm path, physics and host release logic. Its single `controller.run_reference`
call accepts `profile` (the existing bounded time profile) and `contact_feedback`:

```json
{
  "profile": {},
  "contact_feedback": {
    "gain": 0,
    "phases": ["lift", "invert"],
    "loss_threshold_n": 0.1,
    "loss_dwell_s": 0.02,
    "offsets_rad": [[0.01, 0], [0, 0.01]]
  }
}
```

The example is for a two-pad, two-joint ownership map, not a robot configuration.
Use the runtime's pad count, finger ordering, pressure groups and phase labels.
Every correction row may affect only its pad's owned joints. Phases must exist in
the frozen reference. Gain is [0,1], the contact-loss threshold is (0,0.5] N and
dwell is [0.01,0.5] s. Correction amplitude and rate use the trusted
`refinement_bounds`; combined profile/feedback corrections cannot exceed those
bounds. Native joint constraints still apply.

The host observes per-pad simulated forces at each existing feedback tick. After
sustained loss in an allowed phase, it ramps toward that pad's specified posture
offset. Restored contact ramps it back. Missing/nonfinite observations, inactive
phases or substantial table/rack support prevent new seating corrections and
unload existing ones. The supported threshold is half the configured object
weight. Before pressure acquisition and at release the host clears corrections;
the existing release controller retains authority. A conflicting native-limit
and correction-rate requirement stops the trial before another physics step.

With `gain: 0`, the observation-conditioned component is an exact bypass. To
test complete zero correction, also use `profile: {}`. This must reproduce the
retained sequence's states, velocities, commands, timestamps and expected case
outcomes before authoring nonzero candidates through this backend.

This is a declared contact-loss rule, not arbitrary model-generated live Python,
slip/tilt feedback, wrist replanning, residual RL or a hardware controller.
Simulator contact forces and support estimates are privileged observations;
real sensing and reliability remain separate requirements. Unit tests and exact
zero replay do not establish that nonzero corrections improve retention.

For a nonzero trial, record the realized correction alongside per-pad forces,
support, phase and release state at every feedback tick. Report how often it
activated, its peak amplitude, and the first/last activation in both execution
and reference time. Retiming can make those clocks different. Check the complete
saved arrays against the frozen baseline when an outcome is unchanged: a rule
that never activates can preserve nominal success without demonstrating any
recovery capability.

Review the contact state at activation, not just the final success flag. A
single-pad loss trigger can arrive after the other pads have already lost the
object. Brief recovery can also reset the dwell and produce small intermittent
corrections. These are reasons to inspect earlier sliding and load sharing;
they do not by themselves justify increasing amplitude or relaxing limits.
Compare failed-trial metrics over matched windows before calling a shorter or
longer rollout an improvement. Preserve unsuccessful cases and inactive-rule
results in the proposal history. Keep release corrections at zero and retain
the baseline unless the unchanged promotion criteria are satisfied.

Keep trigger selection separate from correction design. A contact-loss event
that occurs only in archived failures can identify a useful intervention window,
but it does not identify a beneficial joint offset. Check a proposed owned-joint
direction against pad-center displacement, normal alignment, native joint
margin and object/environment clearance across loaded snapshots, including
successful cases. An alignment-only fit can move a pad off its support patch.
Archived trajectories that already include an intervention are not untouched
baseline counterfactuals; label their provenance when using them for a new rule.

See [reference profiles](REFERENCE_REFINEMENT.md) and
[author-loop controls](AGENTIC_AUTHOR_LOOP.md). The unchanged admission criteria
and independent held-out evaluation still govern promotion and later training.
