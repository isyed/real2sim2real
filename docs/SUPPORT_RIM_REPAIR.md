# Tabletop repair from visible top and rim observations

Fit a support plane from explicitly selected visible top-depth pixels and test
the fixed plane against separate frames. Keep object and person masks excluded.
A top-only selection cannot establish slab thickness: its depth variation can
be prediction noise. Select the visible vertical rim separately and use
`real2sim2real.retargeting.support_rim.measured_thickness` to measure its depth
below the fitted plane. Check that result on held-out rim observations too.

Use `planar_outline.slab` to intersect the reviewed RGB perimeter with the
accepted plane and construct a convex tabletop. Preserve other scene parts
unless separate evidence supports changing them. Save the coordinate transform,
source hashes, selections, plane residuals, thickness quantile and uncertainty.
The result remains a predicted geometric hypothesis, not measured CAD.

Physical admission is a separate stage. Compare the original setup, support-only
change, mesh-derived inertials alone and their combination with identical object
poses, mesh collision caches and solver settings. Record initial settling and
late drift separately. CPU diagnostics do not replace CUDA validation when the
deployment simulator uses CUDA. A fixed support body does not imply its object
is stable. Uniform-density mesh inertials are a modelling assumption, not the
real object's measured mass distribution. Existing actors need reevaluation
after changes to support geometry or inertials.

Keep three decisions separate in reports: geometric evidence, isolated settling
repeatability, and full-controller task performance. Passing the first two is
not enough to replace an existing policy. A more stable resting object may
respond differently to a hand impact and reduce contact retention. Retain a
successful isolated change as a versioned experimental candidate when the
full-controller result regresses; do not automatically promote it or erase the
older policy's compatible dynamics.
