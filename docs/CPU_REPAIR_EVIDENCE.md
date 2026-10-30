# Exact raster acceleration and retained hypotheses

Dense predicted meshes can repeat hundreds of thousands of integer raster
triangles at a small projected size. The silhouette rasterizer now identifies
identical ordered integer polygons using collision-free packed keys. It preserves
Pillow truncation, winding/order, native grids and the exact triangle union.
Coordinates or point counts that cannot be represented safely take the original
six-coordinate deduplication path. No geometry, topology or acceptance threshold
is approximated. Compare native masks pixel-for-pixel before relying on a timing
change; a faster renderer alone is not a reconstruction improvement.

Changed repair proposals are now written under `proposals/` before scoring.
Their receipts begin with `acceptance_checked=false` and
`full_reference_accepted=false`; completed checks add validation and before/after
scores. A per-frame CSV records the measured image/depth/hand-gap changes.
Rejected and interrupted proposals remain inspectable without replacing the
selected checkpoint or reference. A partial benefit can therefore be retained
without disguising failed or unmeasured gates as acceptance.

## Unit-invariant hand-distance measurement

The point-to-triangle inside test uses a dimensionless barycentric tolerance.
Its orientation numerator has length-to-the-fourth units, so the numeric
comparison scales with the squared face normal. A fixed absolute tolerance
incorrectly treated some small or skinny triangles as much larger plane patches;
the same geometry could report different distances in metres and centimetres.
Degenerate faces still contribute their finite segments/endpoints. The dense
broad-phase bounds use the same dimensionless tolerance as the exact face test.

This changes a measurement, not an acceptance threshold. Historical hand-gap
results must be remeasured before their rejection or acceptance is interpreted.
A corrected hand-distance check alone does not approve an otherwise unvalidated
trajectory. Tests cover analytic tiny triangles, dense/sparse parity, coordinate
unit changes and rigid transforms.
