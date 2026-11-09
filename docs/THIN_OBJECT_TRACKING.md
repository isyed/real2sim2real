# Thin-object tracking and reuse of measured evidence

A depth residual compares a proposed mesh pose with a predicted depth map.
Failure does not establish that depth is the only faulty input. A later clear
RGB registration can give a better pose basin; backward tracking can then
recover frames rejected by forward tracking with the same mesh and depth.

An edge-on silhouette can be only a few pixels wide at a CPU optimizer's reduced
resolution. `LocalRegistrationConfig.optimization_image_max_size` can retain
native resolution for a bounded diagnostic. The CLI exposes the opt-in
`--continuous-image-max-size`; its default remains384. Final GPU acceptance is
always native-resolution. CPU triangle unions and GPU rasterization can disagree
near a threshold: never round a rejected overlap upward to admit a pose.

`recovery_quality.pixel_translation_proposals` generates a finite25-pose grid
within an explicit(0,2]pixel radius, preserving rotation and optical-axis depth.
`--continuous-subpixel-radius` enables this check around continuous CPU proposals;
default0 disables it. Every hypothesis still faces the existing GPU image,
depth, support and motion vetoes. A better overlap does not authorize a temporal
jump. `--continuous-translation-first` optionally fits translation with a fixed
proposed rotation before the full fit, checking the result against the real
previous accepted state. Budget at most two finite fits per recovery call.

The translation-first path and isolated native-resolution/subpixel proposals
have bounded native GPU evidence. The new resolution/subpixel CLI combination
has CPU configuration/gate tests; it is an experimental online option and needs
its own guarded end-to-end trial. Do not describe isolated proposal validation
as validation of every CLI combination.

Accepted forward/backward traces are not automatically a complete reference.
Keep explicit monotone source mappings, retain missing intervals, and check
every join in forward source order. The tracking-evidence helper supports
increasing or decreasing consecutive mappings and rejects repeated/bouncing
indices. Only exact unchanged normal tracker updates retain the normal tracking
gate; recovered/changed poses retain the stricter recovery gate.

Per-frame GPU image/depth/support measurements can be reused only when pose,
mesh, native RGB/depth/object/human-mask payloads, camera convention and support
are verified identical and producer evidence is hash-bound. Reordering does
not change those measured quantities, but changes motion history: recompute all
forward temporal checks against the actual preceding pose with the original
rejection function. Label this as exact GPU measurement reuse with fresh CPU
temporal checks, not a new whole-clip GPU render. Changed inputs invalidate
reuse. Reconstruction agreement remains separate from physical stability,
retargeting, contact and task execution.
