# Bounded prerecorded-reference conditioning

`flash_chord.runtime.actions.reference_denoising` supplies optional temporal
conditioning for explicitly named continuous joints. SONIC's
`reference_denoising` config defaults to `None`. An offline symmetric Gaussian
blend has explicit time windows, strength and maximum angle change. Oversized
changes fail instead of being clipped. Encoder velocities are derived from the
same conditioned positions; filtering velocities alone can create contradictory
motion instructions. The symmetric filter uses an already known demonstration,
never future physical observations.

Use `real2sim2real.controller_actor_training.reference_conditioning.package` to
create an independent policy with preserved source hashes. It refuses existing
body-guidance experiments and retains immutable reference/weight files. This is
an experiment in controller conditioning, not a reconstructed-motion repair or
proof that feet are planted. Actual encoder identity is recorded during physics
and the video labels the conditioned controller.

Useful checks before simulation:

- Distinguish path length from net displacement using explicit timestamps.
- Compare source motion, retargeting and actual physics separately.
- Avoid inferring physical foot contact from a link's height alone. Foot rotation
  and collision geometry affect the support height.
- Check FK, floor clearance, unmodified hand targets and fade-boundary continuity.
- Confirm the disabled path matches the previous encoder exactly.

Then compare matched unassisted physical cohorts. Lower reference velocity or
acceleration is not a task-success result. Require physical approach, balance,
contact and task checks before adopting the variant.

A matched frozen-controller experiment reduced leg-reference velocity and
acceleration but worsened approach and heading. Treat smoother encoder input as
a hypothesis, not a repair certificate: a pretrained controller may depend on
the original position/velocity distribution. Small object-pose score gains can
also reflect an undisturbed object rather than successful manipulation. Retain
per-trial contact and designated-rollout evidence alongside aggregate scores.
If this test fails, investigate source-supported stance timing and jointly
consistent foot/pelvis retargeting before increasing smoothing strength.
