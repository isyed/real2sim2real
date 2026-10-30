# Carrier-guided motion hypotheses across occlusion

`attachment_motion` provides CPU primitives for testing, rather than assuming,
an object/carrier relation. A carrier can be a predicted palm or another rigid
set of corresponding anatomical landmarks. Inputs use explicit metric world
coordinates and original frame IDs. No model, dataset or cloud account is loaded
by these numeric functions.

- `carrier_frames` fits proper rigid transforms using corresponding points and
  reports deformation RMS. It does not fit scale or establish physical contact.
- `diagnose` fits a rigid carrier/object transform on declared fit anchors. A
  disjoint validation set can veto it; validation anchors do not refit the model.
- `diagnose_point` fits only an object-origin point in the carrier frame. It
  deliberately estimates **no object orientation**: wrist rotation need not
  rigidly determine how a grasped object rotates.
- `bridge_positions` preserves independently estimated endpoint positions and
  spreads bounded endpoint discrepancies smoothly while the carrier supplies
  interior motion. The complete interior remains inferred, not observed.
- `regularize_inferred_positions` makes one Gaussian smoothing proposal for
  inferred positions, projects corrections into a per-frame bound, preserves
  endpoints and checks the resulting step bound. It never changes input source
  observations. A smooth output is not an accuracy measurement.

Defaults used in the bounded test were 1 cm carrier deformation RMS, 10 cm
attachment/endpoint consistency, 8 cm frame steps, and at most 3 cm smoothing
correction with a one-frame Gaussian sigma. These are declared hypothesis bounds,
not general physical constants. Failed hypotheses do not replace their inputs.
No function approves a reconstruction, hidden motion, controller training or
physical execution.

## Independent orientation evidence

A translation seed may reuse an old rotation as an initialization assumption.
That rotation is **not** an independently estimated post-occlusion orientation.
`diagnose(..., rotation_observed_frames=[...])` excludes such validation rotations
from the rotation veto while retaining independent position checks. Every rigid
fit anchor must have an independently estimated rotation; otherwise use the
point-only model. A raw angle comparison is retained separately for diagnosis.
The default checks all supplied rotations, so callers must accurately declare
the provenance of their anchors. Symmetry ambiguity also remains a separate issue.

## Validation order

Declare fit/validation frames and any carrier-selection rule before inspecting
validation outcomes. If the model changes after a development failure, record
that retry; disjoint-from-fit does not automatically mean previously unseen data.
If a validation anchor is then used as a bridge endpoint, it is no longer an
independent validation anchor for that **bridge**, even if it was independent of
the original attachment fit.

Check endpoint discrepancy and frame-to-frame motion first, then use independent
visible image/depth/support evidence and occlusion consistency to veto the
hypothesis. An empty mask alone does not prove physical occlusion. A projected
object hidden by a person mask does not prove correct 3D depth ordering. Retain
raw depth conflicts and orientation uncertainty; do not validate an inferred pose
against depth synthesized from that same pose.

A bounded integration test produced a position bridge whose steps stayed under
6 cm after at most 3 cm correction. Independent source images still rejected it
near disappearance/reappearance. This is why numeric continuity alone cannot
approve a hidden trajectory. Reacquire visible segments independently and use
joint image/contact fitting if a fixed attachment cannot explain the data.

Tests exercise exact rigid recovery, independent-anchor rejection, deformation
reporting, disjoint fit/validation, position-only constraints, exact bridge anchors,
bounded smoothing, rejection of large corrections/jumps and exclusion of borrowed
rotations from independent validation. Rendering and downstream execution remain
separate, explicitly labelled stages.
