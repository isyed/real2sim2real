# Data contracts

Every completed stage writes `artifact.json` with schema
`real2sim2real.artifact/1`, a kind, named file roles, SHA-256 hashes and metadata.
Files must reside inside the artifact directory. Consumers verify hashes and
reject missing/changed files. An artifact can be moved without rewriting its
file roles. Model/robot assets remain explicit external dependencies.

## Reconstruction

`geometry.npz` contains numeric arrays only (`allow_pickle=False`):

| Array | Shape | Meaning |
| --- | --- | --- |
| `vertices`, `faces` | V×3, F×3 | Object-local vertices in metres; integer triangle indices |
| `object_scale` | scalar | Uniform scale applied once to the vertices |
| `object_rotation`, `object_translation` | T×3×3, T×3 | Local-to-world transform, column-vector convention |
| `intrinsics` | T×3×3 | Camera intrinsics in native image pixels |
| `world_to_camera` | T×4×4 | Explicit rigid world-to-camera transform |
| `frame_ids`, `timestamps_s` | T, T | Increasing source-frame identity and actual presentation time |

The default RGB adapter uses a static OpenCV camera as its world frame:
x right, y down, z forward. It does not solve arbitrary moving-camera SLAM.
Depth is camera-Z in metres, not distance along a normalized ray. Mesh
generation and object tracking consume the same depth/camera pair; supplied
point maps are checked against the native image grid. Metric depth remains a
prediction, so common scale bias and poor geometry are still possible.

`human.npz` holds native MHR parameters `[T,204]`, shape `[T,45]` and face
`[T,72]`. These are native model parameters, **not a generic metres-valued
joint array**. The MHR decoder and SOMA conversion explicitly handle their
unit/axis conventions and preserve per-frame identity. The human/object
timelines must agree. Model-independent object geometry is not mixed with
human decoder internals.

Other roles are the mesh, source timeline, masks, predicted depth and quality
diagnostics. Backend pickle intermediates are created internally in the run;
the public artifact interface never accepts an untrusted pickle.

## Repair

The repair adapter creates a self-contained numeric bundle for the numerical
engine. Metric evidence is an NPZ with exactly `frame_ids`, `depth_m`, `fit`,
`floor_planes` and `floor_valid`. Plan at least three calibration and three
held-out frames before fitting. Planes use camera coordinates, unit normals
and the oriented floor convention enforced by `metric_calibration.py`.
`floor_roi_source` must identify the reviewed floor evidence.

Calibration is bound to the input and evidence hashes. Fine repair cannot
certify a proposed enlarged mesh without resizing/retracking it and checking
that new input. A failed later pose gate does not delete a valid size
hypothesis. Parent reconstruction, rejected hypotheses, support placement
and actor readiness remain separate records. Human repair is opt-in and
requires a hashed MHR decoder specification and human observations.

## Robot reference

`reference.parquet` uses the included `motion_v1` schema: robot root pose,
joint targets, object pose, end-effectors, source FPS and hand contact
schedule. Quaternions are `wxyz`; the robot world is z-up. Object meshes and
URDF paths are relative to the artifact root. An optional `support.usda`
must already be in the final robot coordinate frame and be self-contained.

Reference contacts are **geometric proximity labels**, not measured physical
contacts. Retargeting is constrained IK, not physics simulation or a learned
grasp policy. The current exporter requires a uniform timeline and rejects
irregular timestamps instead of silently changing timing.

## Controller policy

FlashSAC writes `policy_<step>.safetensors` (inference actor),
`state_<step>.safetensors` (optimizer/learner state), and optional
`replay_<step>.safetensors`. Only policy files are published as actors.
The training artifact records the parent reference hash and command. Retain
the source reference and external controller/robot assets for inference.
Resume training through the trainer's explicit checkpoint/replay options;
pipeline `--resume` skips completed stages and does not restart a partial
training stage automatically.
