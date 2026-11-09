# RoboSimStudio source exchange

Keep RoboSimStudio in its own checkout and virtual environment. This optional
adapter lives in `real2sim2real.rss_bridge`; scenario configuration, RGB data,
candidate selection and competition evaluation belong in a downstream project.
There are no runtime imports of neighboring repositories.

The first experiment should exchange a known reconstruction before modifying it.
This implementation supports **object-pose source inspection**, not a complete
human-to-robot RSS reconstruction system. RSS is not an automatic replacement
for human estimation, segmentation or monocular tracking. No simulator or GPU
is started by exporting/importing a bundle or constructing inspection specs.

## Environment

Install a pinned RSS checkout in a separate environment, then install this
project there with `pip install --no-deps -e /path/to/real2sim2real`. The bridge
requires NumPy, SciPy and trimesh (also used by RSS). Tests additionally need
pytest. Pin the RSS Git revision and freeze the resolved environment alongside
experiment receipts. Do not upgrade an already validated training environment
to accommodate RSS's Newton/Warp dependencies.

## Geometry contract

Input NPZ fields:

| Field | Shape / convention |
|---|---|
| `vertices`, `faces` | Mesh-local vertices in metres; triangle indices |
| `object_rotation`, `object_translation` | `(T,3,3)`, `(T,3)`; object to source world |
| `object_scale` | One positive, explicit scalar |
| `intrinsics` | `(T,3,3)`; native-resolution, zero-skew camera matrices |
| `world_to_camera` | `(T,4,4)`; CV camera: right X, down Y, forward Z |
| `frame_ids`, `timestamps_s` | Complete ordered IDs and timestamps in seconds |

The caller supplies image dimensions and one rigid `source_to_rss` matrix.
No scale, depth, camera, frame-rate or origin is guessed. Export preserves the
original mesh and creates a PLY from the trajectory-local vertices; this
avoids importing hidden GLB scene-node transforms. Mesh ordering/topology are
verified and preview quantization is recorded and limited to 0.1 micrometre.
The PLY preview has no source
texture. The human asset is opaque and hash-preserved, with its own declared
coordinate frame; it is **not rendered as if aligned** to the object.

Historical float32 rotations may have orthogonality drift up to `1e-5`.
RSS uses quaternions, so exchange rotations are projected to the nearest proper
rotation; the source matrices remain byte-preserved. The receipt reports both
the measured drift and a conservative bound on mesh-vertex displacement from
this representation change. Larger drift, reflection and scale are rejected.
Incoming edited poses must meet the stricter `2e-6` rigid-transform tolerance.

## CLI

```bash
python -m real2sim2real.rss_bridge export \
  --geometry geometry.npz --mesh object.glb --human human.npz \
  --width 1536 --height 864 --source-frame camera_world \
  --human-coordinate-frame aligned_human_world \
  --transform-json source_to_rss.json --rss-revision PINNED_COMMIT \
  --rgb source.mp4 --output exchange
python -m real2sim2real.rss_bridge import \
  --bundle exchange --poses exchange/poses_rss.npz --output identity_roundtrip
python -m real2sim2real.rss_bridge check-camera \
  --bundle exchange --frames 0 100 200 --output camera_parity.json
```

For actual edits, copy `poses_rss.npz` outside the immutable exchange bundle.
It contains `frame_ids`, `timestamps_s` and `object_to_rss` matrices only.
Import rejects partial, reordered, retimed, nonfinite, scaled or reflected
poses and detects modified source assets. An unchanged round trip copies the
geometry **byte for byte**. Edited imports change only object poses and retain
unchanged frames exactly. Every return is a new artifact with explicit
`source_validation=false`, `physical_validation=false`, `selected=false`.

## Remaining stages

The camera check exercises the installed RSS camera API, including nonidentity
transforms and off-centre intrinsics. It does not validate dynamics. Support
planes, human alignment, material/physics properties, robot retargeting and
controller behavior require separate verified adapters or the existing pipeline.
Keep the same robot, frozen controller and evaluator when measuring the effect
of a source repair. A kinematic pose preview is never a physical task rollout.

## Native CPU source inspection and stationary intervals

`rss_bridge.native_render.SourceRenderer` builds the exact-mesh RSS scene on
CPU and renders cropped, native-resolution RGB/segmentation/depth through RSS's
tiled renderer. Poses are explicitly written for reconstruction measurement;
the resulting images **are not physical rollouts**. RSS's automatic ground at
z=0 can hide camera-space objects below that height. The inspection adapter
uses a separately named synthetic cell with its floor at -100 m and unchanged
object/camera transforms. That floor is never a calibrated task support.

`rss_bridge.rest_pose.fit_pose` fits one bounded pose across multiple observed
masks for an interval already judged stationary from source RGB. It retains
mesh and camera, limits evaluations/time and can lock camera-centroid depth.
Silhouette-only fitting can otherwise move an object toward the camera and
improve overlap while worsening depth consistency. Validate predicted depth
independently, and reject such a proposal without weakening acceptance limits.

`blend_stationary_tail` makes an explicit smooth transition and a fixed tail.
That interpolation is an inferred repair, not measured hidden motion. Validate
all changed frames, source support constraints and temporal seams. A stationary
image does not establish stable physical contact, mass or friction.
