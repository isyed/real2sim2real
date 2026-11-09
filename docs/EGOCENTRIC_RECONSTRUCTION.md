# Calibrated egocentric reconstruction

Install `modules/egocentric-reconstruction` for moving-camera rigid-object work
and `modules/reconstruction-evaluation` separately for CPU reference scoring.
The existing full-body reconstruction and G1 retargeting paths remain independent.

`calibrated-capture` -> `egocentric-reconstruction` -> `object-reconstruction`

The new artifact is intentionally not accepted by the full-body retargeter:
object-only inference cannot stand in for decoded human motion. Later hand
reconstruction must add explicitly typed hand geometry, identity, validity and
world transforms before a hand retargeter can consume it. Likewise, no policy
success is implied by reconstruction scores.

Camera geometry uses right-handed OpenCV axes (right/down/forward), metres and
column-vector transforms. `rig_from_camera` maps a stream into the common capture
rig; `world_from_camera` maps each target camera into the fixed first-camera
world. Object world poses are the product of `world_from_camera` and
`camera_from_object`. Video pixels are undistorted onto one declared pinhole grid.
The resized intrinsic uses pixel-centre scaling. All depth is camera-Z, not ray
length. Invalid stereo observations remain missing.

The source capture records video hashes, consecutive frame IDs and timestamps.
Do not replace capture time with a container FPS hint. Private dataset adapters
must resolve vendor camera ordering and image orientation before creating this
generic contract. Calibration fits and held-out frames must be disjoint; the
baseline's known metric length is fixed during relative-pose refinement.

The initial portable object artifact contains:

- `geometry.npz`: metric `world_from_object`, `world_from_camera`, pinhole
  `intrinsics`, `frame_ids`, `timestamps_s`, and `accepted`.
- A triangle mesh in exactly the body frame used by `world_from_object`.
- Camera/tracking quality reports and a source timeline.
- A standard hashed `artifact.json` receipt with kind `object-reconstruction`.

Intermediate `object_motion.npz` preserves both accepted and raw rejected poses.
Rejected poses never seed subsequent tracking. An incomplete run can be diagnosed
and scored with explicit coverage, but cannot publish a complete accepted artifact.

Reference scans and trajectories live only in the evaluator. Mesh registration
may estimate similarity for geometric correspondence; trajectory translations
remain metric and are never rescaled. World alignment happens once at the first
reference frame. Missing predictions reduce coverage and conservative AUC; they
cannot turn into successful episodes by silently shortening the evaluation.

GPU execution uses `real2sim2real.runtime.container_guard`, which controls the
owned container itself. A separate privileged `temporary_gpu_clock.py` helper
enforces the bounded lease and restores clocks. Source, model stores and one
prediction directory are explicit mounts. Do not mount a project root containing
ground truth as the prediction directory. CPU-only import checks precede GPU runs.
