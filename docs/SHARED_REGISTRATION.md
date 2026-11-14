# Shared RGB/stereo registration and timing hypotheses

`real2sim2real.egocentric.shared_registration` fits one bounded rigid correction
to an existing source-to-target camera transform. `fit_shared_registration`
consumes only explicitly supplied training correspondences, uses a robust 1.5 px
loss, and bounds each rotation-vector and translation component. It returns a
separate hypothesis with convergence/bound diagnostics; no calibration artifact
is modified. Stereo depth errors and unknown exposure offsets can bias this fit.

Declare training landmark subsets and held-out views before fitting. Save the
candidate before scoring unused views. Report both absolute residuals and
relative change from the original transform; improvement on a training view
does not establish cross-view transfer. Preserve rejected corrections and original
source/image/calibration hashes. Previously inspected views are only held out
from the new fit, not fresh independent evidence.

Timing requires a separate relative-motion model. Absolute per-offset static PnP
can absorb registration error and make a timing curve change for the wrong reason.
`fit_motion_from_flow` estimates camera motion using **image displacements**:

```
project(M * X) - project(X) = observed_target_pixel - observed_anchor_pixel
```

The absolute registration stays fixed during each offset sweep. Zero displacement
produces identity motion, even when anchor projections have a fixed bias.
`flow_errors` evaluates withheld static displacement residuals. Use the same
common marker identities at every offset, and keep depth/annotation uncertainty
and appearance-track limitations visible. A good flow fit is not proof of metric
camera calibration or a correct physical marker center.

If a shared rigid correction improves one view and worsens others, inspect
intrinsic/distortion/image-orientation conventions, stereo depth and timing
before accepting a more flexible correction. Rotation metadata alone does not
prove the images require a second rotation; verify the released-pixel convention.
Do not fit calibration to a reconstructed object merely to lower its residuals.

Tests check cross-view recovery of a synthetic shared transform, identity motion
with biased anchor projections, and held-out flow prediction with known camera
motion. They do not establish hardware calibration or synchronization.
