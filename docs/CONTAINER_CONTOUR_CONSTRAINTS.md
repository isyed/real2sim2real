# Visible container contours and diameter constraints

`real2sim2real.egocentric.container_contours` adds explicit boundary evidence to
the exterior-profile fitter. It addresses a weakness of containment-only image
checks: a projected model can contain observed blue pixels yet have the wrong
width. Distances from observed boundary pixels to the corresponding projected
boundary penalize both undersized and oversized shapes.

Each annotation declares a name, `kind` (`bottom`, `rim` or `silhouette`),
`visibility` (`visible`, `occluded` or `unknown`), image pixels, positive
`sigma_px`, and whether the complete end contour is observed. Coordinates must
match that view's calibrated image grid. Keep disconnected visible side segments
as separate entries; never join a boundary through an occluding hand.

- A **complete bottom/rim contour** with at least six nondegenerate observations
  supplies fitted ellipse major/minor diameters in pixels. Both observed-to-model
  and model-to-observed boundary residuals are included, along with diameter
  differences. Diameter uncertainty is twice the declared boundary uncertainty.
- A **partial visible arc or side segment** supplies observed-to-model boundary
  distances only. It does not produce an ellipse diameter or fill a hidden arc.
- An **occluded/unknown entry** produces no residual and is marked unavailable.
  Missing evidence is not counted as a passing observation.

Projected end rings use the current profile radius, height, camera pose and
intrinsics. Pixel diameters are not independent metric diameter measurements.
Boundary annotations may follow a physical lip absent from the current model;
keep that modeling discrepancy explicit. Annotation uncertainties are assumptions
unless independently measured, and assistant-selected points are not independent
human ground truth.

`constraint_report` returns per-entry availability, contour residuals, measured
image ellipse parameters and projected diameter errors. `contour_residuals`
returns the fixed-length normalized vector for optimization. Supply annotations
as `image_constraints` on a training view passed to `container_profile.fit_shared`.
Existing views without this field preserve their previous objective. The fitter
does not consume validation annotations unless the caller improperly places
them in its training set; dataset adapters must enforce the declared split.

Before refitting, audit annotations against images and evaluate fixed hypotheses.
Comparing both profiles at a common pose helps distinguish shape-width changes
from pose/depth compensation. Also retain each hypothesis's original pose as a
separate diagnostic. Do not modify the existing acceptance criteria to match the
new objective. Used development views cannot later be claimed as untouched
validation views. No contour agreement establishes wall thickness, contact
support, protrusion clearance or control readiness.

Tests cover projected diameter recovery, penalties for both shrinkage and growth,
unknown/occluded evidence handling, partial-arc semantics and synthetic fitting
through the optional contour-constraint interface.
