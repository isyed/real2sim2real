# Retaining a grasp during demonstrated object motion

`controller_actor_training.grasp_retention` adds bounded translation feedback to
an established grasp while keeping its independent object-motion target fixed.
It is an optional controller for the checked Cartesian wrist fixtures used by
`object_grasp_pilot`, not a general arm controller or a learned policy.

At hold entry it records the object-origin vector in the primary wrist frame.
Subsequent error is the difference between the observed world-space vector and
that same fixed vector rotated by the current wrist. The controller integrates
a speed-limited proportional correction and bounds its total world-space offset.
Common rigid object/wrist motion creates no correction. Commands go through the
existing force-limited robot position actuators; the object remains unactuated.

The explicit `grasp_retention` configuration contains `gain_s_inv`,
`max_speed_m_s`, `max_offset_m`, `minimum_contact_n` and
`maximum_reference_error_m`. All must be finite and positive. Updates require
both thumb and opposing-finger normal forces above the declared minimum and
object/reference position error below the declared maximum. Otherwise the
existing offset freezes. Joint-limit violations reject the rollout. The fixed
anchor never resets to hide accumulated drift. Per-step update, freeze and
saturation counts are recorded alongside the world offset.

Feedback uses exact simulator object/wrist poses and contact forces. A hardware
implementation would need a validated observation path. Wrist-relative object
displacement is a retention diagnostic, **not measured sliding at finger pads**.
A wrist can follow a drifting object: retain separate object-motion fidelity,
loaded-contact, support, orientation and penetration checks. Motion fidelity is
relative to the anchored reconstruction and does not supply independent ground
truth. Saturating a correction is evidence to inspect contact layout, not an
automatic reason to increase its travel limit.

## Matched experiments

```bash
python -m real2sim2real.controller_actor_training.retention_experiment \
  --config /data/retention.json
python -m real2sim2real.controller_actor_training.retention_review \
  --comparison /data/retention-run/comparison.json \
  --output /data/retention-run/review
```

The experiment accepts a hash-pinned `parent_trial`, `parent_evaluation`,
`sources`, fresh `output`, finite `deadline_s`, one to three named `controllers`
with explicit `settings`, and one to four named `validation` cases containing
`hold_s`, `mass_scale` and `friction`. It reuses the parent's scene, staged grasp,
fingers, capture trajectory, timing, force limits and acceptance criteria.
The uncorrected baseline must reproduce its saved states exactly. Selection
requires a complete physical and reference-fidelity pass; the lowest primary
slip among development passes is frozen before matched validation. Validation
cannot shorten the original hold. Failed cases remain in the comparison.

The CPU worker has a finite deadline. No GPU, external repository imports,
training or automatic promotion occurs. A passing declared experiment still
does not certify arms, real sensing or hardware. The review renders saved states,
checks their hashes, includes portable WebM/GIF playback, and lists physical and
reference outcomes separately. Identical state archives may share one video.

See [guided motion](GUIDED_MOTION.md), [wrist coordination](WRIST_COORDINATION.md)
and [testing](TESTING.md).
