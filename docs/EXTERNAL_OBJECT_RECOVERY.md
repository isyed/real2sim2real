# Optional external object recovery

This is an alternative **tracking component**, not a replacement for the default
reconstruction pipeline or a reproduction of the full Do as I Do system.

## Installation boundaries

The CPU diagnostic `reconstruction_repair.scale_depth_audit.audit` tests whether
one common metric scale can reconcile declared same-pixel surface-depth
observations across frames. Supply an explicit finite grid containing unit
scale and keep every depth hypothesis separately labelled with provenance.
Scaling camera-space object points changes mesh dimensions and translation
together while preserving their image projection. Therefore a good silhouette
cannot establish metric scale. This diagnostic writes no repaired geometry and
does not check human contact, support, motion or external seams. A failed grid
rules out only those scales for the fixed orientations/surface correspondences;
it does not rule out other shape or pose repairs. Human-aligned depth, a mesh
generated from that depth, and a floor fitted from it are dependent evidence.

The optional `reconstruction_repair.joint_rgb_pose.fit` is a CPU alternative
for proposing object orientation and position jointly across a short window.
It uses a fixed metric mesh, calibrated cameras, fit-frame outline observations,
occlusion flags, predicted support and temporal regularization. Depth is absent
from its fitting interface. Withheld query times provide no image measurements;
their interpolated poses remain hypotheses until independently checked.

Declare trusted knots with `locked`. If the anchor itself is uncertain, an
explicit `allow_unanchored=True` trial may re-estimate it, with incoming/outgoing
seams still unvalidated. Up to seven deterministic orientation restarts share
one time/evaluation budget and one set of bounds relative to the original seeds.
They never accumulate displacement by reusing a fitted seed. Selection uses fit
cost only. Final checks must use native triangle silhouettes, each declared depth
hypothesis, support and motion limits; a better convex outline or optimizer
convergence is not acceptance. The mesh size and camera are never optimized.

Keep [Do as I Do](https://github.com/malik-group/do-as-i-do) in a separate checkout.
Follow its reconstruction setup for the Fast-SAM3D submodule, compatible SAM3D
environment and pretrained weights. This repository does not redistribute those
models or upstream implementation. SAM3/HaWoR are not required by this adapter
when callers provide their own RGB, masks, fixed mesh, camera and source depth.
The prepared adapter targets upstream revision
`824591b808c342b20079c3b4198a8c2bdf88c74e`; requests additionally pin the actual
tracking entry-point files because submodule contents can differ.

`real2sim2real` provides:

- `reconstruction.daid_recovery`: an explicit-file adapter invoking the external
  tracker, retaining all 25 sampled poses and the optional refined winner.
- `reconstruction.hybrid_tracking` and `conditioned_recovery`: normal tracking
  first, external proposals only on failure, followed by independent acceptance.
- `reconstruction.recovery_lookahead`: opt-in offline admission that also requires
  successful subsequent normal tracking; preserves rejected-branch diagnostics.
- `reconstruction.daid_shape`: CPU input/coordinate qualification and supervised
  GPU shape encoding for a new fixed metric mesh.
- `reconstruction.daid_inputs`, `daid_render`, `daid_progress`: shared-camera
  pointmaps, parity-checked full-viewport raster binning and actual-work progress.
- `reconstruction_repair.canonical_pose` and `fixed_scale_pose`: checked coordinate
  conversion and coupled scale/translation handling for a fixed physical mesh.
- `reconstruction_repair.joint_depth_pose`: optional joint re-estimation of
  uncertain anchors and latent foreground depth from multiple predictions.

Do as I Do therefore remains a runtime dependency for **new external proposals**.
Reading and independently checking saved proposals does not require its models.
Normal tracking remains the default; no pipeline setting is silently changed.

## Portable recovery request

Run within the upstream-compatible environment, with this repository's `src`,
`modules/reconstruction/src` and `modules/reconstruction-repair/src` on `PYTHONPATH`:

```sh
python -m real2sim2real.reconstruction.daid_recovery --request request.json --cpu-check
```

The JSON request explicitly supplies absolute `checkout`, `config_dir`,
`input_dir`, `output_dir`, optional `vendor_dir`, `source_frame`, `previous_frame`,
`previous_pose_camera`, `depth_hypothesis`, `source_video_sha256`, and `mesh_sha256`.
An `accepted_state_path` plus `accepted_state_sha256` binds the conditioning pose
to a receipt with `source_frame` and `pose_camera`; rejected states cannot be
substituted. GPU inference additionally requires `accepted: true` and nonempty
`validation_provenance` from independent validation; a CPU preparation receipt
alone is insufficient. Inputs use source-frame IDs, camera-frame rigid poses, optical-axis
depth in metres and the original pixel grid.

Required input members are `canonical_mesh.obj`, metric `shared_mesh.obj`,
`shape_latent.pt`, `anchor_layout.json`, and `NNNNNN.png`, `NNNNNN_mask.png`,
`NNNNNN_HYPOTHESIS.npz`. The NPZ contains `source_frame`, `K`, and float32 `depth_m`.
The layout supplies `K`, canonical axes `Q`, normalization and `center_z`.
`input_sha256` pins each member. `upstream_sha256` pins `track_object.py` and
`sam3d_objects/pipeline/inference_pipeline_pointmap.py` relative to Fast-SAM3D.
Model paths are supplied by the caller's reviewed pipeline configuration.

CPU checking exercises actual preprocessing, pose-target latents, canonical
mesh/pose round trips and model configuration imports. It is not GPU inference.
Under a separate thermal/time/memory supervisor, use `--run-gpu` instead of
`--cpu-check`. The adapter itself is not a background scheduler or thermal guard.
Use sequential model lifetimes on memory-constrained GPUs. Model setup, sampler
inference and parity validation still require the upstream dependencies.

The current bounded settings retain 25 samples, 25 diffusion steps, chunk size 4,
shape guidance 1 and fixed pose guidance 0.5; no TAPIR or shape ICP. All raw samples
are exported alongside the post-optimized winner. Each actual run checks native
versus binned rendering agreement before using the optimization. A sampler's
image score never approves a reference. Validate the final fixed-scale export.

## Depth disagreement and validation

`joint_depth_pose.fit` accepts only declared fitting-frame observations, initial
camera poses, a fixed metric mesh and requested timestamps. It fits movable pose
knots plus latent foreground depth constrained by every named depth prediction,
image outline, support and temporal priors. `refine` can refresh the fitting-frame
front-surface correspondences after rotations, under one bounded CPU budget.
Both return hypotheses; interpolation between knots is explicitly inferred.

Do not validate fitted poses against their own latent depth. Independently check
unchanged source predictions, full triangles, visibility, support, motion and
withheld frames. Moving an earlier anchor also requires checking its join to the
previous interval. A smooth path is not measured hidden motion. Model-derived
depth and a held-out consistency pass do not establish true metric scale or
physical grasp feasibility. Failed hypotheses remain separate from accepted
references. Episode-specific media and experiment orchestration belong downstream.

The reusable adapter and joint fitter have focused CPU tests. A downstream
experiment must still establish native GPU validity and task-specific benefit;
these modules do not promise improved reconstruction or success rates.

An individually accepted recovery can still make subsequent normal tracking
worse. Compare complete-window coverage and image overlap on the same accepted
frames, not just the number of sampler proposals that pass. Before production
adoption, test short look-ahead branches from candidate recoveries and require
successful handback to normal tracking with depth and motion checks. The opt-in
`recovery_lookahead.qualify` now implements that bounded branch selection;
`conditioned_recovery_native` accepts a `lookahead` request with consecutive
`frames`, `horizon` (1–5) and `max_candidates` (1–5). It checks the best qualified
current-frame candidates, rejects any branch failing a future depth/motion/image
gate, and returns the selected normal-tracker updates with their evidence.
The native hook currently supports the unchanged `original` depth hypothesis.
This uses future observations and must be described as offline tracking.
The native hook has been exercised on GPU, including rejecting single-frame
passes whose next ordinary tracker update violates the motion bound. This is
evidence that the admission check executes, not that hybrid tracking improves
full-window coverage. The current focused suite has 47 passing CPU tests across
look-ahead, conditioned selection, hybrid tracking, shared inputs and the
external request contract. Expanded diagnostic fields are CPU-tested separately
from the immutable GPU experiment snapshot.
A sampler's final
refined winner can fail motion or depth even when raw samples pass, so retain
and independently evaluate both. Preserve the original reference when the
window-level comparison regresses.

For a new fixed mesh, `daid_shape --cpu-check` validates canonical coordinates,
identical voxel occupancy, actual image/point-map preprocessing and configuration
imports using the separate upstream installation. The supervised GPU mode
encodes its shape once, separately from proposal inference. CPU checks do not
claim that the encoder or diffusion model has run. Never reuse a shape cache
from a different object merely because the model configuration is the same.

## Reading a look-ahead receipt

For example, a caller can add this to an original-depth native selection request:

```json
"lookahead": {"frames": [101, 102, 103], "horizon": 3, "max_candidates": 3}
```

The selection in this example is at source frame 100. Future IDs must be
consecutive and available in the native input window. The callback must start
each branch from its exact candidate pose and then use its newly accepted poses.

`accepted` at the top level is final admission. A row in `candidates` with
`accepted: true` only passed the current-frame checks. The receipt distinguishes
`current_frame_accepted`, `evaluated_future_frames`, `untested_qualified_candidates`
and `selected_updates`. Each trial records `step_diagnostics`, including measured
evidence even when a motion veto prevented its use for acceptance. Those
measurements are never a substitute for passing the veto. All future steps must
pass; a truncated or gapped window cannot approve a recovery. Ranking is bounded,
so rejection does not imply that every possible candidate was tested.

Compare against a fresh ordinary-tracker pass with the same initial pose and
inputs. A saved repaired reconstruction can have a different history and is a
display baseline, not necessarily the result of that fresh tracker pass. Check
coverage first, then overlap on common accepted frames, depth, temporal motion
and hand alignment. If the return to normal tracking repeatedly fails, examine
foreground depth and the tracking update itself before sampling more independent
poses or weakening limits. Neither source consistency nor smoothness establishes
ground-truth motion or physical task success.

## Optional depth-guided ordinary updates

`reconstruction.depth_guided_update.correct` can repair translation after a
normal RGB tracker proposal fails. It preserves the proposal's rotation and the
metric mesh. From the signed visible-surface residual against the unchanged
original depth, it generates three camera-ray translations, bounded by an
explicit maximum displacement from the failed proposal. The final displacement
from the **last accepted pose** must still pass the original temporal limit.
These two bounds have different origins and are recorded separately.

The native `conditioned_recovery_native` request enables this explicitly:

```json
"depth_guided_translation": {"max_move_m": 0.15, "target_residual_m": 0.03}
```

The option works for an ordinary `track` request and for the future updates in
an offline `lookahead` selection. It uses only the declared `original` depth,
requires sufficient visible depth support, and applies the stricter recovery
image gates to corrected proposals. Good original updates are retained exactly.
All proposed corrections and independent measurements are recorded. Failure does
not fill a pose or update the retained state. No mesh resize, rotation repair,
synthetic depth or threshold relaxation is performed.

When enabled during look-ahead, the future tracker is **normal tracking plus
bounded depth correction**; do not report this as successful handback to the
unmodified normal tracker. Test the new sequence from a common starting pose.
Cached failed updates may have different predecessor poses and elapsed times,
so individually passing CPU replays do not establish a stable new trajectory.
Depth remains a monocular prediction, and native GPU and temporal validation are
required. This option remains disabled by default.

The opt-in path has now executed under native GPU validation and improved a
short sequence only partially. The remaining failure still failed the original
depth check; it was not silently accepted. In mixed-sign surface residuals, a
median signed correction can stop short even when a broader bounded ray search
has CPU-feasible hypotheses. Such a search is a separate declared experiment,
requires native validation and must not be reported as part of the completed run.

A subsequent single-proposal native check accepted a preselected bounded ray
translation from the exact previously accepted state. This supports testing
finite ray proposals when the signed-residual heuristic stops short; it does
not validate an automatic expanded search or longer tracking. Preserve the
accepted-history identity and distinguish reused validated prefix frames from
new measurements when composing a short result. The default three-proposal
correction algorithm remains unchanged.

### Explicit finite translation search

`reconstruction_repair.depth_translation.finite_ray_proposals` generates up to
64 unique, explicitly listed signed Euclidean translations along the mesh-centroid
camera ray. Negative values move toward the camera. The mesh and rotation remain
fixed, the correction-distance bound is enforced before proposal generation, and
centroids crossing the camera plane are excluded. No candidate is approved by
this geometric helper.

`reconstruction.depth_guided_update.finite_ray_search` measures those proposals
using a supplied independent measurement callback and the existing recovery
selector. It checks motion against the **last accepted pose**, then image,
absolute depth discrepancy and support against the declared original source.
All candidates and failures are retained. This is an explicit separate experiment;
it is not enabled by the native tracker's existing three-proposal option.

```python
result = finite_ray_search(
    failed_pose, accepted_previous_pose, vertices=metric_vertices,
    source_frame=frame, previous_frame=previous_frame, fps=fps,
    signed_distances_m=[-i * 0.005 for i in range(31)],
    max_move_m=0.15, measure=measure_original_source,
)
```

Choose the search direction from declared source evidence; do not change the
depth map to favor a fit. The default selector ranks passing proposals by IoU.
If a downstream experiment instead preselects by absolute-depth margin, record
that decision and preserve the full search evidence. A CPU pass still requires
native validation from the same accepted predecessor. A single passing frame
does not establish stable tracker handback; continue only after validation and
stop at the next rejected update. The limits remain unchanged in either case.

In a native follow-up, an interior finite-ray proposal passed independently but
the next ordinary update still failed temporal motion. The signed depth residual
also reversed direction, and its heuristic corrections worsened that motion.
This is why finite-ray acceptance remains local evidence: check temporal depth
consistency and later tracker behavior before treating it as a full repair.
Common-image-pixel depth changes are a useful warning, but do not establish
ground-truth motion because occlusion and projected surface identity can change.

`recovery_quality.consistency` now also records signed residual quantiles
(10/25/50/75/90 percentiles) and positive/negative fractions. Its depth-error,
signed-median and acceptance calculations are unchanged. Use this distribution
to distinguish a coherent offset from a broad mixed-sign discrepancy. A signed
median can point in a direction that worsens the accepted-state motion and does
not minimize the median absolute discrepancy after reprojection. Compare
explicit bounded candidates against the actual gates; do not infer correction
direction or certainty from the signed median alone. A near-threshold pass with
large residual tails should remain labeled uncertain.
# Spatial holdout for depth experiments

`reconstruction_repair.foreground_depth.spatial_floor_holdout` checks a fixed
depth calibration on reviewed floor rectangles that do not overlap its fitting
regions. It reports coverage and residuals without fitting another scale.
The optional `floor_holdout_rectangles_xyxy` probe setting applies this check
to both full-frame and camera-centered crop predictions; a failed holdout blocks
depth qualification. Existing requests without this setting retain their behavior.

A checkerboard split inside a fitting patch is useful but samples nearly the
same surface. A separate spatial patch can expose local disagreement that split
misses. Neither check proves foreground accuracy: the floor plane is predicted,
neighboring pixels are correlated, and full/crop inference uses the same model.
Keep original depth and accepted poses immutable. Compare foreground changes
before and after diagnostic floor calibration; do not automatically replace depth
with whichever hypothesis makes the tracker pass.

Also report cross-scale disagreement when both variants are temporally smooth.
`audit_probe` deliberately returns no preferred variant, and `conditioned_recovery.select`
requires every declared depth hypothesis to pass. Test original and diagnostically
floor-calibrated controls separately. A finite translation search that fails does
not prove no rotation/shape/pose solution exists; it establishes only that the
tested depth-only corrections cannot reconcile the evidence.
