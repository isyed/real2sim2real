# Reconstruction

This module contains the existing RGB segmentation, human initialization,
metric depth, object generation and pose-tracking implementations. The new
stage adapter connects them in this order:

1. SAM2 segments the person and manipulated object from explicit prompts.
2. MoGe2 predicts depth and camera intrinsics; SAM 3D Body estimates native
   MHR motion. Depth is aligned to that predicted human geometry.
3. The aligned native-resolution depth is unprojected with the same camera
   into a metric point map for SAM 3D Objects.
4. SAM 3D Objects generates the mesh; its predicted scale is applied once.
5. Ordinary FoundationPose tracks that mesh using the same aligned depth
   and camera, with bounded registration/recovery and quality diagnostics.

The important integration correction is **shared metric input before mesh
generation**, rather than generating a mesh in a separate depth frame and
trying to fix all size discrepancies afterward. This reduces an avoidable
inconsistency; it does not make monocular metric predictions ground truth.
The default stage performs initialization, not a full joint refinement.

Use `examples/reconstruction.json` and provide a real prompt file. Prompt
object ID 0 is the person and ID 1 is the manipulated object. Example prompt
coordinates are illustrative and must be replaced for your video. Current
scope is one visible person, one rigid manipulated object and a static camera.
There is no automatic support-surface reconstruction in this first adapter.

## Backend environment

The build recipe is `docker/Dockerfile`, using the repository root as context.
Choose and record explicit upstream commits for separately installed SAM2,
SAM 3D Objects and nvdiffrast. The recipe is supplied for reconstruction
environment preparation; **it has not been rebuilt and GPU-validated as part
of this extraction**. Existing compatible environments can also install
`pip install -e . -e modules/reconstruction`.

```bash
docker build -f modules/reconstruction/docker/Dockerfile \
  --build-arg SAM2_REF=YOUR_REVIEWED_COMMIT \
  --build-arg SAM3D_REF=YOUR_REVIEWED_COMMIT \
  --build-arg NVDIFFRAST_REF=YOUR_REVIEWED_COMMIT \
  -t real2sim2real-reconstruction:local .
```

Provide models in the explicit layout in [INSTALL](../../docs/INSTALL.md).
All model acquisition happens outside a run. The stage requests offline
Hugging Face operation. If you split SAM2/SAM3D and tracking into independent
environments, `backend_pythons` maps `segment`, `depth`, `human`,
`depth_alignment`, `metric_mesh`, or `object_motion` to their Python
executables. Install core and reconstruction distributions in each one.

Run with a guarded GPU session only after CPU preparation. The new stage
adapter's contract is CPU tested; end-to-end model inference in this packaging
still needs a short acceptance clip.
