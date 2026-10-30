# Bounded initial-pose fitting and partial improvements

`real2sim2real.reconstruction_repair.initial_pose_fit.fit` refines up to three
initial SE(3) hypotheses on CPU using native-grid triangle silhouettes, predicted
surface-depth correspondence and explicit camera-frame support evidence. Its
inputs are a fixed metric mesh, rigid hypotheses, intrinsics, boolean object and
human masks, predicted metre depth and support provenance.

```python
from real2sim2real.reconstruction_repair.initial_pose_fit import fit
result = fit(vertices_m, faces, initial_poses_camera, intrinsics,
             object_mask, human_mask, predicted_depth_m, support)
```

The default bounds are 180 seconds, 900 assessed candidates, 0.25 m translation
and 60 degrees rotation around each initial hypothesis. Rotation preserves the
mesh center while translation moves that center. These are declared initial
uncertainty bounds, not inter-frame motion tolerances: frame zero has no invented
previous accepted pose. Mesh size, shape, input depth and existing acceptance
thresholds remain fixed. Bounds and measured runtime are retained in the result.

Every measured hypothesis is recorded. If any passes all CPU checks, the highest
IoU valid hypothesis is retained even if an invalid one has a lower optimization
cost. Otherwise the best cost is returned explicitly as rejected, with individual
metrics and rejection reasons. Neither case approves a reference or physics;
independent renderer validation and full-timeline evaluation remain necessary.

A rejected proposal can still improve some criteria. Preserve it separately and
report image/depth/support outcomes individually, rather than either discarding
all information or promoting it to an accepted trajectory. Better agreement with
predicted depth/floor is not proof of true geometry. A one-frame improvement is
not a repaired motion sequence. A finite search failure does not prove that no
valid pose exists.

When fixed-mesh pose fitting improves depth/support but image agreement remains
poor, inspect shape, source-view occlusion and metric evidence. A clearer RGB
view can help generate shape, but its monocular depth must still be calibrated
independently. A source occluder mask can be used for an optimistic explanatory
bound; it must not silently replace depth-validated occlusion handling or relax
the original image gate.
