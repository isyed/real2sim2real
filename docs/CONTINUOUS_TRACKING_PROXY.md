# Continuous object-tracking proxies

`continuous_tracking_proxy` evaluates a completed arm replay against its own
reconstructed object-motion reference. It does not estimate error against an
unavailable ground-truth reconstruction. Keep its result separate from physical
acceptance, independent robustness trials and official benchmark scores.

```bash
python -m real2sim2real.controller_actor_training.continuous_tracking_proxy \
  --config /data/tracking-proxy.json
```

The configuration contains `result: {path, sha256}` and a fresh `output` directory.
The input must be a demonstrated continuous physics replay with saved states,
original frame IDs and native capture timing. Stationary diagnostics are excluded.
The runner verifies the arm-audit/source-trial ancestry, reference, task, mesh and
state receipts. It uses the saved object poses, not the commanded wrist poses.
NumPy, SciPy and trimesh are required; simulation and GPU dependencies are not.

Position interpolation and quaternion Slerp sample observations at the reference
timestamps. Missing coverage is rejected. When the first observation is exactly
one physics step after initialization, the pinned replay source supplies the
actual initialized object pose; it is not extrapolated from later observations.
The reconstructed motion increments retain the frozen simulator resting anchor.
There is no fitted trajectory alignment, time warp or extra independent lift.

The report uses object ADD AUC over 1–9 cm thresholds, six attached ±5 cm object
keypoints for MPPE, and SP-SR/MP-SR pose thresholds. These metrics do not measure
finger articulation. The implementation uses the existing vendored evaluation
helpers and records their hash. The single-object RPE value is zero by convention;
it provides no evidence about multi-object relative motion.

The reconstructed area-weighted surface centroid defines the local body center.
A second result exposes sensitivity to using the native mesh origin. A seeded
sample of 500 reconstructed mesh vertices supplies ADD. These choices cannot
replace official scanned geometry, body registration, allowed symmetries or
ground-truth poses. Mesh CD-O remains unavailable. The report explicitly labels
`official_score=false` and `reconstruction_error_included=false`.

SP-SR and MP-SR from one trajectory are threshold outcomes for one world, not
estimates of generalization. Preserve the original physical rejection reasons
and release verdict alongside them. The added release/settling tail is outside
the native-duration pose metric and must pass its separate physical checks.

## What the continuous protocol means

Acquisition establishes the grasp through approach, pad seating and finger
closure. The object then follows its demonstrated lift/motion/return on the
original capture clock. Once the table supports it, the hands open, unhook and
withdraw; a terminal interval checks stable placement and release. A diagnostic
stationary hold freezes a suspended command to isolate retention and is a
different experiment. Its success cannot stand in for full demonstrated motion.

## Residual learning

The current `ArmResidualEnv` supports staged replay and explicitly rejects the
continuous protocol. Before extending it, preserve a named full-motion comparator
and an independent stationary development baseline. Implement phase-aware
observations/rewards and verify complete zero-action equality, including return,
release and physical verdicts. Do not silently reuse a staged hold target for
continuous rewards.

A bounded residual pilot can investigate recovery from a rejected baseline; a
baseline pass is not a mathematical prerequisite for residual RL. First establish
that bounded actions affect the failure without breaking other gates. Pressure
authority alone does not prove authority over wrist slip or lift clearance.
During carrying, reward object tracking and retention; after table-supported
placement, reward unloading and release instead of sustained contact. Keep
collision, force, actuator and correction bounds independent of reward weights.
Use held-out payload, friction, pose and observation perturbations to measure
generalization. Relative improvement on the development trajectory never changes
the original acceptance limits.
