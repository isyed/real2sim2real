# Observed scene screening before grasp fitting

Use this CPU stage to assess calibrated stereo observations before fitting robot
hands or completing hidden surfaces with a learned mesh. It reconstructs partial
object/support/obstacle surfaces in a common metric frame and tests local contact
evidence. It does not require an existing object mesh, reference poses or scans.

`real2sim2real.egocentric.observed_scene` provides the geometry functions;
`real2sim2real.egocentric.scene_screening` orchestrates them and exports an offline
review. Both belong to `egocentric-reconstruction`; there is no dataset dependency.

## Inputs and execution

1. Build a hash-bound `calibrated-capture` manifest with actual capture timestamps.
   Verify stream frame counts, released image conventions and stereo ordering.
   Inspect measured capture time separately from nominal dataset timestamps and
   video presentation timestamps; they may have different spacing and offsets.
   Preserve the chosen field and its provenance. If this changes, version the
   capture and dependent outputs rather than silently retiming a frozen baseline.
2. Run the existing `calibrate`, `prepare` and `camera` phases. Choose a static
   background ROI and exclude moving subjects. Use disjoint calibration frames.
3. Annotate polygons in the **rectified left stereo image**, not the RGB grid.
   Define the object, contact subset, visible support and nearby obstacles.
   These image annotations must not be selected using favorable depth residuals.
4. Choose a short interval in which the selected surfaces are stationary. Check
   image-only mask propagation in every exported overlay. Object motion requires
   a separately validated object transform; this stage must not fuse moving parts
   as static geometry. Background odometry is not drift-free ground truth.
5. Pin the capture, calibration and camera-motion files. Start from
   [the configuration example](../modules/egocentric-reconstruction/examples/scene-screening.json).
   Every new hypothesis needs a fresh output directory and a finite CPU timeout.

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  timeout 600 python -m real2sim2real.egocentric.scene_screening --config CONFIG
```

Source pins are rechecked after execution, and implementation hashes are saved.
Only the declared observations are read; the command does not load a model or
access a GPU. Fit and held-out frames must be disjoint and match the capture's
timestamps and accepted camera poses.

## What the checks mean

- **Depth coverage:** dense SGBM is checked against a separately computed reverse
  disparity map and local image texture. Raw and filtered observations are both
  retained. Missing or inconsistent pixels never become geometry. A separate
  sparse NCC search samples the image region before consulting dense depth;
  it searches the full configured disparity range with ambiguity and round-trip
  gates. Both methods share the same images and calibration, so they are not
  independent ground truth.
- **Metric surfaces:** triangles connect only adjacent valid pixels and have a
  bounded metric edge length. No hole filling or hidden-surface completion is
  performed. Voxel thinning keeps actual measured points rather than averaging
  across surface gaps. Point counts across frames are correlated, not counts of
  independent measurements. PLY exports are partial surfaces, not closed solids.
- **Table:** fit a robust plane on declared fit frames and validate on other
  frames. The upright transform assumes the selected support is level. The
  separate slab is an explicit hypothesis: its thickness is configured and its
  convex footprint can span occluded tabletop. Do not silently treat it as a
  measured complete table. Below-plane outliers remain reported.
- **Contact patches:** check an example 4 mm pad radius against observations in
  a 12 mm neighborhood, including 2D coverage, 1 mm p90 surface residual and
  normal repeatability over at least three frames. These are example screening
  settings, not a claim about a particular robot pad. The first local normal is
  estimated from points; other frames corroborate it. No independent true normal
  or material thickness has been supplied.
- **Initial access:** sample a straight cylindrical finger approach away from a
  patch, compare it with the table and reliable stereo free space. Unknown depth
  stays unknown. A sample behind the observed surface is occluded, not proof of
  solid occupancy. Even a completely clear sampled probe does not certify the
  complete swept volume, articulated finger, palm, opposing grasp or arm path.
- **Uncertainty:** half-pixel depth sensitivity excludes calibration and systematic
  correspondence errors. Held-out-to-fit nearest-point distances are repeatability
  diagnostics, not object-surface accuracy. Calibration bound hits remain visible.

`summary.json` always declares `control_ready=false` and `physics_run=false`.
Passing a plane or a few patches does not admit a grasp. Continue through the
separate gates in [CONTROL_READINESS.md](CONTROL_READINESS.md): initial hand fit,
insertion and reachability checks, appropriate stationary holds, then loaded motion.
If finite contact evidence is missing, preserve the partial scene and investigate
geometry before spending a control-proposal budget. A completed parametric or
learned mesh must remain a separately labeled hypothesis until corroborated.

## Review and checks

The offline HTML includes a rotatable metric point cloud, left/right images,
region/depth overlays and PLY downloads. Serve it with the loopback review server;
the server supports PLY downloads and video byte ranges, while retaining its
hidden-file and outside-root restrictions. No CDN or remote rendering is used.

```bash
python -m real2sim2real.review_server --directory /data/runs --port 8774
pytest -q tests/test_observed_scene.py tests/test_egocentric_geometry.py \
  tests/test_stereo_contact_evidence.py tests/test_review_server.py
```

Tests exercise reciprocal matches, missing-depth visibility, rigid-frame
composition, depth-discontinuity meshing, source immutability, metric probe bounds
and media path restrictions. They do not validate a physical grasp.
