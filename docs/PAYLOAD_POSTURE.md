# Supported-load posture transitions

`payload_posture` adds an optional bounded finger-posture transition to the
bilateral fixture controller. It complements pressure adaptation when one fixed
posture works at nominal/heavy load and another works at lower load. This is one
observation-driven controller, not a choice of successful trials by case name.

The pure `BoundedPostureFraction` consumes estimated supported weight divided by
a declared reference weight. Above `activation_weight_ratio`, it requests the
original posture; below `full_weight_ratio`, it requests the admitted alternative.
It interpolates between those ratios and rate-limits the fraction using both its
fraction-speed cap and the largest finger-joint displacement. Missing supported
load freezes the fraction. Simulator mass and case labels are not policy inputs.

Enable `contact_force_feedback.posture_transition` alongside `payload_adaptation`
in a downstream `bilateral_evaluate` configuration. For example:

```json
{
  "target_result": "/data/admitted-alternative/candidate-1/result.json",
  "target_sha256": "<sha256>",
  "activation_weight_ratio": 0.95,
  "full_weight_ratio": 0.8,
  "maximum_joint_change_rad": 0.35,
  "maximum_joint_speed_rad_s": 0.2,
  "maximum_fraction_speed_s_inv": 1.0
}
```

These are experimental bounds, not universal robot settings. Actual endpoint
displacement can be smaller than the travel budget. The posture speed bound
applies to its contribution; the bounded pressure servo adds its own correction.
Combined commands still respect native joint and actuator control ranges, with
clipping counted in telemetry.

The current `PayloadPosture` adapter supports two Cartesian wrist fixtures only.
It requires a pinned, complete physically/reference-admitted nominal endpoint,
matching compiled scene, unchanged planning penetration criterion, same joint
ordering, fixed wrists and immutable object targets. Only finger joints already
owned by the pad controller may change. A sampled path check rejects native-limit
or collision failures before physical acquisition. Intermediate pad points are
projected onto existing compiled collision hulls with a 1 mm correction bound.

This projection constructs controller sites, not observed contacts or collision
geometry. Sampled checks do not certify continuous clearance or loaded stability.
The adapter does not move the physical object, reset grasp anchors, change wrists
or weaken force limits. Absent this option, the original control path is retained.
A zero fraction returns its input finger command unchanged.

Per-sample `contact_force_control.payload_posture` receipts record fraction,
desired fraction, actual fraction-rate bound, endpoint displacement, update/freeze
counters, clipping, endpoint hash and static path audit. Load estimation uses
privileged simulator contact forces and vertical acceleration, not RGB-derived
weight or a hardware sensor implementation.

Admit a complete stationary hold, then nominal demonstrated motion with identical
settings and unchanged criteria. Pin that result and repeat the entire declared
matrix through `grasp_sensitivity`, requiring identical nominal replay. Evaluate
transient acquisition and drift as well as the final posture. Do not pool cases
from different rules. Loaded-arm validation remains a separate requirement.
