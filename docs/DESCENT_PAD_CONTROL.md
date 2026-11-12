# Descent pad seating and two-hand load sharing

`descent_pad_control` is an optional phase-scoped policy inside the existing live
finger-pressure owner. It can be evaluated against an original continuous arm
replay or an exactly pinned wrist-feedback comparison. It does not add actuators,
object forces, artificial contact constraints or reconstructed geometry.

The replay configuration contains:

```json
{
  "descent_pad_control": {
    "baseline": {"path": "/data/comparison/result.json", "sha256": "..."},
    "activation_start_s": 4.0,
    "controller": {
      "normal_labels": [],
      "anchor_labels": [],
      "anchor_settings": null,
      "rebalance_n": {},
      "share_gain_n": 0,
      "maximum_share_offset_n": 0.5,
      "maximum_offset_n": 1.0,
      "maximum_offset_rate_n_s": 2.0,
      "ramp_s": 0.4
    }
  }
}
```

Choose and declare activation from the known reconstructed trajectory. The first
`return` phase can occur substantially after actual downward motion starts;
inspect reference vertical velocity rather than equating that label with onset
of descent. Activation must follow acquisition. The complete reference, clocks,
acquisition scoring anchor, pressure-servo budgets and physical gates remain
pinned. The same declared activation time is used for matched stationary holds.

Three effects can be tested independently:

* `normal_labels` selects admitted `primary:group` / `secondary:group` pads that
  may use measured object-contact normals/points during the phase. Actual loaded
  contact remains required. This avoids freezing a loaded pad solely because
  contact has migrated away from its original planned point. Lost-contact pads
  retain the original planned-surface geometry guards.
* `anchor_labels` plus existing `ContactAnchors` settings latch measured material
  points once at activation/contact availability. Bounded tangential finger tasks
  seek those points. The points never re-latch to erase later slip. These are
  control anchors; the independent acquisition-time scoring anchor is unchanged.
  Alternatively, optional `zero_tangent_labels` selects observed-normal pads
  whose added pressure-correction task requests zero tangential velocity. This
  constrains pressure-induced scrubbing without following a moving material
  point. These labels must be disjoint from material-anchor labels and a subset
  of observed-normal labels. Original finger rate/travel bounds still apply.
* `rebalance_n` requests pad-specific pressure redistribution. `share_gain_n`
  shifts opposing-finger pressure between hands according to their measured
  positive upward support fraction. Signed forces are retained in telemetry;
  negative upward force is not treated as positive support. Sharing requires
  isolated hand support and freezes when observations are invalid/table-supported.

Only five anatomical groups per hand are admitted. Rebalance must sum to zero
over the pair. The cross-hand sharing direction also sums to zero. A common
interpolation factor preserves this additional pair sum during pressure slew.
Combined per-pad offsets are bounded to 1 N, with explicit rates. Existing payload
adaptation still establishes nominal goals; final original per-pad goal caps can
limit the redistributed goals, and clipping is recorded. Conservation of the
additional offsets does not imply identical measured contact forces or unchanged
state-dependent payload estimates across different physical rollouts.

Pressure corrections stop acquiring new targets at capture end and recover
smoothly to zero by scheduled release even after contact loss. The return interval
must support recovery within the rate bound. Material tasks stop with table/other
support or release; the source supported opening and withdrawal remain the owner.
Stationary diagnostics retain the phase policy throughout the full hold with
release disabled. No shortened scoring window or reset of an acceptance anchor
is allowed.

Source result/scene/state hashes and all base settings are verified. Wrist feedback
and this finger policy may compose only when that exact comparison is pinned.
Existing tangent controllers, loaded-posture transitions and pressure actors may
not share finger-task ownership. Original joint travel/rate limits, collision path
screening, native physical penetration checks and wrist-feedback command guards
remain. Actual loaded trajectories and release still require complete evaluation.

Always run empty labels/anchors/rebalance and zero sharing first against each
comparison controller. Compare every saved physical state, diagnostic record,
scene, physical verdict and continuous/release metrics. New telemetry is stored
separately in `descent-control-trace.json`; it does not rewrite existing records.
Results declare `promotion_eligible=false` and `training_run=false`.

Useful diagnostics include pad force versus target, normal source, frozen-update
counts, tangent gap, material tangent velocity/friction utilization and signed
bilateral support through the same descent interval. More pad pressure does not
by itself prove a better load path. Compare measured-normal, seating-only,
sharing-only and combined policies, then complete stationary tests. Keep failed
variants and prior baselines intact.

Closer scalar normal-force tracking is insufficient evidence of improved load
support. If coupled tasks reduce normal authority or every redistribution worsens
retention, audit the available contact forces/moments under friction and actuator
limits before expanding the search or training an actor. A geometric wrench
fingerprint alone does not certify force equilibrium with finite finger torques.
