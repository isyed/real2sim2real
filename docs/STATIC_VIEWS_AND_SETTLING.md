# Static-view depth and object-settling diagnostics

Metric depth aligned independently to a moving human can drift between frames
even when the camera and floor are stationary. High silhouette overlap alone
does not detect that failure. Before retracking, use the reconstruction-repair
module's `static_floor_sequence.inspect_static_patches` on independently reviewed
background regions. Low texture, image displacement or disagreement between
patches must block the static-view assumption.

`fit_sequence` checks every raw depth frame against one original predicted-floor
anchor. Preserve the exact original frame-zero gain. `depth_sequence.materialize`
keeps its depth bytes unchanged, bounds subsequent gains, and requires a complete
timeline without interpolating rejected frames. Neither operation certifies
absolute metric accuracy or approves old object poses; independent fresh pose
validation remains necessary. A longer accepted tracking prefix is evidence of
partial progress, not permission to publish an incomplete repaired sequence.

The controller module's `object_stability.settling_summary` distinguishes late
motion from initial gravity/contact settling using saved rigid-body states. It
reports final-window displacement, rotation and speeds alongside the probe's
whole-interval maxima. The caller owns admission thresholds and must retain the
initial-pose and source-quality warnings. A stationary final state can be a
toppled object, so late drift alone is insufficient. The helper never moves an
object, changes the reference, or labels a result as successful manipulation.

Keep reconstruction quality, physical stability and bounded research admission
as separate recorded decisions. A diagnostic experiment with explicit modelling
warnings must not silently become a certified reconstruction or deployment.
