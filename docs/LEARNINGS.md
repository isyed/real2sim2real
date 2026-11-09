# Module learning register

Record reusable findings here. Scenario identities, private videos, measured
results and deployment settings belong in the downstream experiment repository.
An implementation change is not evidence of improved reconstruction or robot
performance until the corresponding acceptance test has run.

**Audit metric identifiability before repeated scale retries:** camera-centred
scaling of object geometry and translation preserves perspective projection.
Silhouette improvement alone therefore cannot establish scale or depth. Trace
the prediction dependencies: a human used to align depth, a mesh generated from
that pointmap and a floor fitted from it are not three independent metric
checks. The optional `scale_depth_audit` keeps all frame/hypothesis pairs on a
declared finite scale grid and reports common compatibility and residuals
without changing inputs or selecting a replacement. Compare its unit-scale
residuals with the actual tracking metric before interpreting the diagnostic.
If no common scale works, preserve the visually improved research candidate and
seek different evidence rather than choosing the smoothest prediction or
silently increasing tolerances. This does not prove every possible repair fails.

**Separate RGB pose fitting from disputed depth validation:** a short joint
orientation/position fit can improve withheld-frame silhouette overlap while
still failing original depth or alternative depth hypotheses. The optional
`joint_rgb_pose` fitter deliberately has no depth-fitting input, keeps metric
mesh/camera fixed and requires explicit fit-only observations. Its bounded
orientation restarts share the total work budget and original seed bounds.
Previously accepted anchors can become disputed under new evidence: test their
re-estimation only in an explicit unanchored hypothesis, never by silently
unlocking them. Preserve external-seam uncertainty and all rejected depth checks.
Retain a visual improvement as research evidence without promoting it as a
validated trajectory. Previously inspected frames withheld from a new fit are
not a previously unseen benchmark.

**A smooth depth prediction can still disagree in absolute position:** test
foreground depth separately from floor calibration. Keep spatial floor holdout
regions outside all fitting patches, and compare full-frame/camera-centered crop
predictions using the same metric conventions. Same-model crops are correlated
hypotheses, not independent ground truth. Passing floor checks and lower temporal
jitter do not justify choosing a crop that moves the foreground substantially in
depth. Check fixed poses against each hypothesis and retain all failures. If a
bounded search has no common solution, report that result and its limited search
space; do not blend the predictions, relax thresholds or promote the smoother
one. A diagnostic floor-calibrated control must not overwrite source depth.

**Recovery must survive the return to ordinary tracking:** an independently
valid object pose can still produce a large translation on the next ordinary
tracker update. The opt-in `recovery_lookahead.qualify` tests bounded branches
from already qualified candidates and requires every consecutive future update
to pass the original image, depth, motion and support checks. A high image
overlap or single-frame pass cannot commit a branch. Reuse the exact accepted
updates; rejected branches never change the retained state. This consumes future
source observations and is an offline method. It is not a promise of better
whole-video coverage.

**RGB refinement can drift along the camera depth axis:** the RGB-only proposal
mode deliberately zeros observed depth/point-map features inside the refiner,
then checks predicted depth afterward. A strong silhouette score therefore does
not guarantee metric stability. The opt-in `depth_guided_update.correct` adds
bounded camera-ray translations after a failed update, preserving its rotation
and the physical mesh. Recheck every corrected pose against the actual accepted
history and all original image/depth/support gates. Limit both the correction
from the raw proposal and motion from the last accepted pose; one is not a
substitute for the other. Keep the original update and diagnostic evidence.
Success on cached individual updates is insufficient: run the corrected tracker
sequentially, including its future continuation. Do not silently change the
default refiner or treat predicted depth as ground truth.

**Keep failed-handoff evidence distinct from accepted evidence:** cheap motion
gates can reject a proposal even when a native callback has already rendered it.
Look-ahead receipts now preserve those measurements as diagnostic evidence,
record translation and rotation steps, distinguish single-frame success from
final admission, and list actually evaluated future frames and untested qualified
candidates. Rejection still applies; unavailable frames and untested branches
must not be presented as failed measurements. Fixed-mesh encoding is object
specific: `daid_shape` checks canonical/metric coordinate agreement and creates
a hash-bound cache for that mesh before invoking an external proposal model.
The separate upstream installation and its licensing remain required.

**Preflight optional visualization dependencies in the parent:** requesting
display-only mesh decimation requires `fast-simplification` in the rendering
environment. Check it before worker creation and video encoding so a missing
package produces a direct actionable error instead of a broken process pool and
an empty MP4. The `visualization` extra declares these CPU dependencies. Keep
failed render artifacts, reuse independently verified reconstruction results,
and retry only the presentation stage. Seven focused renderer tests pass,
including missing-package failure before any output is created.

**Preserve hand proximity during temporal pose fitting:** unconstrained smoothing
can reduce object jumps while pulling its surface away from the reconstructed
wrists. The opt-in `interaction_translation` helper uses nearest surface witnesses
and hard translation/landmark-gap bounds, with unchanged rotations and explicit
locked frames. Verify exact final surface distances independently. A conservative
fit toward the original translations can preserve image agreement better than
strong acceleration smoothing toward a previously rejected proposal. Neither a
successful optimizer nor a geometric witness certifies physical grasp. Retain
image/depth failures even when only one frame narrowly exceeds a threshold.

**Preserve support evidence across CPU/GPU checks:** predicted camera-frame
support planes may vary across source frames. The native sequence validator now
accepts explicit source-indexed support records, rejects missing/duplicate IDs,
and hashes the support evidence. Never substitute the first plane for the entire
window just to fit a constant-plane API. Existing explicitly constant support
inputs retain their behavior. Nineteen focused CPU tests cover the new proposal,
native frame mapping and support selection; actual GPU validation is separate.

**Long-object rotation and disconnected recovery:**
`elongated_shape.principal_axis` retains the mesh-local major axis once;
`orientation_audit` compares its native-camera projection with the visible
object-mask axis and reports projected-center error separately. Exclude hidden
and near-end-on views. An unoriented image line cannot establish head/tail,
axial roll, metric size or a valid 6D pose. Four focused tests cover these
ambiguities, translation sensitivity and input preservation.

A fresh image/depth-valid registration can still lie outside the motion bounds
of the previous accepted pose. Keep registration validity, within-window tracking,
and whole-reference seam validity separate. A bounded CPU refinement of a failed
fresh-registration hypothesis may improve native silhouette/depth agreement;
revalidate it with the actual GPU renderer before tracking. Retain short recovered
intervals and their failed next frames without filling the gaps. Compare foreground
depth discontinuities with static background evidence before applying any global
depth correction. Videos of partial results must mark unresolved frames and
invalid joins, and must not inherit the original full reference's approval.

**Resume measured tracking prefixes with explicit source identities:** a failed
later update does not invalidate every earlier accepted pose, but an available
state array is not independent quality evidence. `tracking_resume.verified_prefix`
checks hashes, rigid consecutive previous/selected state chains and declared
forward/backward source mappings. It rejects accepted states after failure and
never fills a rejected tail. Tracker-local frame zero may correspond to a later
native video frame; record both rather than labeling a window index as a source
index. A revised endpoint must be rechecked against its own image/depth and the
preceding accepted pose before resumption. CPU look-back proposals and actual GPU
validation remain separate, and a longer prefix does not approve a full reference
or physical grasp. Hash-tamper, chain-break and source-mapping tests cover this
contract; the focused continuation/local-recovery suite passes 22 tests.

**Placement tracking and grasp order:** correct masks do not prevent a 3D tracker
from staying at an old height. Keep alternate admissible orientations for
symmetric objects, lock trusted anchors, and check depth and whole-clip joins
after any image-based correction. Higher mean IoU alone can hide depth failures
and orientation flips. Label uncertain poses in complete videos. A robot that
falls before contact phases requires reaching/balance and reference-timing
diagnosis before finger tuning. See [the reusable placement workflow](PLACEMENT_TRACKING.md).

**Contact manifolds before geometry edits:** a flat object can rock or slide on
a mesh support even when its placement and friction appear valid. Compare an
analytic primitive and an equivalent triangle mesh on both the ground and the
actual support. Test explicit multiple-contact CCD with identical geometry and
poses before clipping a collision base or rotating a reference. The runtime and
isolated tester now expose this opt-in mode, preserving the false default.
Match solver settings, retain short-pass/long-fail results, and confirm on the
intended backend before robot evaluation. See [the diagnostic recipe](MULTICONTACT_SETTLING.md).

`settling_compatibility.compare_solver_settings` checks recorded solver settings
against the controller's physical timestep, ratio, iterations and explicitly
recorded contact mode. A stress test at one solver ratio is not matching-policy
evidence for another ratio, even if both are valid experiments. Missing settings
fail the comparison. Seven focused tests cover mismatches and missing evidence.
Matching settings alone never certify backend, geometry or full-task equivalence.
Retain slight full-rollout initial-rest gate failures separately from tiny late
drift and improved task scores; a useful competition baseline with warnings is
not a strict stability or actor-training admission pass.

**Contact labels and task rewards:** hand-only contact counters and force-closure
rewards do not measure foot contact or validate a non-hand manipulation task.
`reward_alignment.audit` checks reward/effector mismatches independently of
reference contact admission. Its optional non-hand profile removes only the
hand-grasp term from a fresh, consistent curriculum; it neither invents a foot
reward nor changes existing defaults. Preserve object/body rewards and record
the profile before starting a fresh learner. See [non-hand reward auditing](NONHAND_REWARDS.md).

**Static-support topology and initial-rest repairs:** isolated physics checks
must inspect solver topology, not only support transforms copied back for
display. A free-root kinematic table can move internally in a test without the
runtime's reset/pinning lifecycle. The standalone settling tester now attaches
the same imported support shapes to the static world and records support mode,
backend and body/joint counts. Historical kinematic diagnostics remain available
and retain their failed measurements. Static support alone does not repair an
unstable object orientation.

When source images establish an initial resting phase, `support_seed` can propose
bounded mesh-derived support orientations and apply a local rotation about an
explicit centre, with declared hold/fade frame IDs and exact later poses. These
functions generate hypotheses; image/depth acceptance, packaged world-geometry
equivalence, fresh contact labels, repeated CPU/GPU settling and unassisted robot
execution remain independent checks. A stable scene can still have no grasp or
task success, and higher tracking AUC can coexist with worse position error.
Preserve both outcomes when deciding whether to train. See
[the workflow and limits](STATIC_SUPPORT_SETTLING.md).

Validation: the focused support, resting-pose, settling, contact and controller
suite passes 20 tests, including an actual CPU physics run. Downstream acceptance
also completed three CPU and three GPU settling repeats and three unassisted
robot trials per original/repaired cohort. This establishes the bounded tested
case; it does not establish automatic repair success on arbitrary videos.

| ID | Finding | Module change | Evidence and state |
| --- | --- | --- | --- |
| R-001 | Independently installed perception models may require incompatible environments. Selecting subprocess interpreters only for the GPU calls still leaves heavy export/packing imports in the parent environment. | Reconstruction now dispatches every phase, including packing, metric input preparation, object export and artifact publication, to an explicit worker environment. `backend_commands` accepts argument arrays for externally supervised container wrappers; `backend_pythons` remains supported. | CPU dispatcher tests and offline imports in separate segmentation, mesh and tracking environments pass. Acceptance passed on a verified full-clip continuation and a second fresh end-to-end clip, including all-frame tracking and artifact hash validation. |
| R-002 | A partially completed model chain must not be confused with a complete reconstruction. | Record the executing phase, argv, start/finish times, failure type and per-phase logs. Stop at the first failure; refuse to overwrite a partial output directory. | Tested with a failing depth worker: later phases never start, and reuse of the partial directory is rejected. |
| R-003 | Shared depth before mesh generation removes one inconsistent input path, but monocular scale remains a prediction. | Preserve the common camera, native depth grid and point-map validation in reconstruction. Compare independently the resulting size, image agreement, motion continuity, support and human proximity before selecting repair or retargeting. | Camera/depth round-trip and mismatched-camera rejection are CPU tested. Generalization across additional complete videos remains pending. |

For future entries, include the failure mechanism, minimal reusable change,
regression test, measured acceptance evidence, limitations and affected module.
Keep rejected ideas visible rather than promoting them to defaults. A successful
reconstruction check does not establish contact, physical stability or readiness
for actor training.

**Flat-contact solver controls:** an ideal box can be stable on the ground yet
rock on a static mesh table with the same timestep and materials. In a bounded
CPU comparison, explicit multi-contact generation stabilized both primitive/mesh
controls and an unchanged reconstructed object; clipping the object's collision
base was not required for that result. Keep the geometric diagnosis separate
from contact-generation behavior. `SimConfig.enable_multiccd` remains an explicit
boolean opt-in with a false legacy default. `solver_ablation.package` creates a
new frozen-policy artifact with only this setting changed and preserves source
hashes, reference, geometry, inertia and controller. Thirteen focused tests pass.
Repeated CPU settling is not GPU or grasp certification: isolated and robot
solver parameters must be disclosed, and intended-GPU/robot checks still decide
adoption. See [the control workflow](STATIC_SUPPORT_SETTLING.md).

**R-004 — extraction completeness:** excluding every directory named `data`
removed SAM3D Body's transforms and batch-preparation Python code. Restore that
six-file package with its original license headers and provenance. Full backend
CLI import checks (not just wrapper imports) are now part of acceptance. A data
storage exclusion must distinguish source packages from datasets. Validation:
all six full backend CLI imports now pass offline with GPU access hidden. The rebuilt reconstruction wheel contains the restored package.


**R-005 — tracker validity is not reconstruction quality:** saved raw predictions
can mark most frames as tracker-reliable while their projected mesh overlaps the
observed object very poorly. Keep runtime validity separate from independent
image/depth, metric-size, support and continuity evidence. Downstream acceptance
reports now retain each measure separately. Do not select a reconstruction from
`tracking_reliable_frames` alone. No new automatic acceptance threshold has been
promoted into the reconstruction algorithm from this observation.

**R-006 — model-family cache isolation:** a global human-model `TORCH_HOME`
selected a DINO repository without the object model's required pretrained weights.
An offline mesh worker failed after the human and depth phases had completed.
Object generation now receives its own HF/Torch cache directories. Regression
coverage checks the routing for every GPU phase. Both required object DINO models
were loaded from their correct local cache on CPU with networking/GPU disabled.
The preceding inference outputs were preserved and hash-verified for a downstream
continuation. GPU mesh generation, full-clip tracking and complete-artifact publication have passed on a verified continuation; a second fresh end-to-end clip has also completed successfully.

The backend now exposes `--check-backend`: this imports the real model CLI with
CUDA hidden and requests help, catching transitive package omissions before
inference. It is a dependency check, not a complete model-weight or GPU test.

**R-007 — preflight must not alter inference state:** dependency checks now avoid
creating depth/mesh output directories. The check-only path is covered for all six
model phases, including hidden CUDA and no inference output. This keeps a failed
or repeated preflight from obstructing a later run.


**R-008 — image agreement does not certify 3D continuity:** a full regenerated
mesh can fit the observed silhouette well while its inferred depth jumps between
adjacent frames. A visible-size hypothesis may also remain below one because a
projected silhouette and visible-surface depth do not measure the full 3D object
diameter. Preserve metric, silhouette, temporal, support and human-alignment
checks separately. A passed metric diagnostic permits testing a hypothesis; it
never approves a resize or a robot reference. The current calibration contract
already records `approved_for_use=false`, requires fresh tracking after a resize,
and documents foreshortening and surface/center-depth limitations. Acceptance
review now exercises those distinctions on full new reconstructions. No new
automatic resize or smoothing default has been introduced from this observation.


**R-009 — visualization must not hide retained human geometry:** a keypoint-only
comparison can make a complete parametric human reconstruction appear incomplete.
The new `human_surface.decode_surface` helper decodes the retained MHR parameters
on CPU into full-resolution, full-timeline surface vertices and model faces. It
records native-to-OpenCV unit/axis conversion and content hashes, and uses bounded
batches with a memory-mapped array. Coordinate/transform rejection tests pass;
a real two-frame model decode produced finite forward-facing surfaces with
18,439 vertices and 36,874 faces per frame. Complete source timelines are now
decoded in downstream comparisons; final video verification is recorded there.
The display does not repair pose or establish grasp/contact readiness.

The surface helper also verifies an already-completed cache without importing
Torch. This lightweight path rejects changed input/model or output bytes; the
focused surface suite now passes three tests. The active downstream inference
snapshot remains immutable and records the earlier helper implementation.


**R-010 — regeneration must compete with a retained strong parent:** fresh raw
outputs can improve over a tiny original yet regress against an already repaired
reference. The downstream experiment fixes candidate hashes before regeneration
and evaluates both geometries on identical source frames/masks. It preserves
historical physics evidence rather than imputing it to a new reconstruction.
Reconstruction now publishes a separate `quality` report of producer image
agreement and temporal warnings, with explicit unassessed dimensions and no
readiness approval. All-frame tracker reliability is retained separately. The
new report is CPU-tested; the active inference snapshot is unchanged, so new
publication-path GPU acceptance remains for a subsequent run. No model default,
mesh size, trajectory or training reward was changed from these observations.


A repeated fresh reconstruction with nearly unchanged mesh dimensions can still
produce worse tracking. Diagnose image agreement, depth drift and temporal jumps
before requesting another mesh resize. Keep unsuccessful calibration hypotheses
and their failed validation evidence, so regeneration does not accidentally
reintroduce a previously rejected change. These findings support the existing
separate gates; they do not establish a new automatic correction rule.

**R-011 — a good resized seed may still float:** foreground-depth registration
can improve silhouette overlap while placing a resting object above its inferred
support. Keep resting clearance separate from nonpenetration. The optional
`support_seed.propose_resting_seed` helper requires an explicitly reviewed resting
frame and tries bounded plane-normal translations, retaining actual-parent image,
depth and support vetoes. It returns the original seed on rejection and never
propagates a resting assumption to the trajectory. Synthetic tests cover bounded
selection, parent preservation, insufficient correction bounds and mandatory
phase evidence. A downstream predicted-data seed passed the checks after a small
translation; full tracking and physics validation remain pending.

**R-012 — standalone tracking must share the pipeline's explicit model runtime:**
a direct vendor CLI invocation can import successfully yet search for checkpoints
relative to source code. The reusable `tracking_runtime.tracking_environment`
checks both score/refiner configurations and checkpoints, supplies the explicit
weights directory, offline flags and bounded render/refiner batch settings. The
normal object-motion backend uses it too. Checkpoint/configuration presence is
not a model-loading test: downstream acceptance separately loaded both actual
checkpoints on CPU and checked finite tensors before requesting a GPU retry.

**R-013 — report paired gate measurements, not just failure labels:** improving
average silhouette fit or smoothing a trajectory can still damage previously
good frames, predicted-depth agreement or reconstructed hand separation. Gate
reports now retain the actual maximum paired hand-gap increase and its threshold,
explicitly distinguishing reconstructed hand points from physical contact. The
acceptance rule is unchanged; historical immutable runs retain their original
reports. A rejected temporal proposal must never replace its tracking parent,
even when it greatly reduces acceleration. Keep size/support hypotheses and
local motion acceptance separate, and mark experimental tracking candidates as
review-only when global checks fail.

A downstream full-clip tracking execution completed using the bounded resting
seed and explicit runtime. It improved average image fit, while local regressions
and discontinuities remained. This validates the launch path, not reference or
actor readiness. No new automatic smoothing, resizing or gate relaxation follows
from that result.

**R-014 — validate prompts against the actual RGB before model execution:** saved
segmentation boxes can be numerically valid yet point at the wrong subject. A
wrong person mask can later fail human-anchored depth correspondence, which is a
symptom rather than permission to lower the correspondence threshold. The video
phase now creates a native-resolution first-frame box/point preview and a hashed
prompt-review receipt before segmentation. The reusable helper rejects invalid
coordinates and out-of-timeline prompts; it explicitly leaves semantics unverified
and identifies later-frame prompts that need their own review. CPU preflight
callers should inspect this preview before launching model phases. A downstream
RGB-only prompt correction is retained separately from its failed attempt; the
active frozen inference snapshot remains unchanged.

**R-015 — few jumps can coexist with an entirely wrong trajectory:** a fresh
multi-sequence run completed and published quality reports, but one near-zero
image-agreement trajectory had fewer discontinuities than a much better tracked
sequence. A smooth or slowly drifting prediction is not accurate motion. Keep
image agreement and continuity as independent findings; repair recommendations
must not rank solely by jump count or average over a good initial segment.
Visible-depth calibration can also become unassessed when occlusion leaves too
little reliable evidence. Preserve that result rather than inferring a resize
from the remaining frames or allowing continuity to override missing evidence.
The existing quality flags expose these cases; no acceptance threshold changed.

**R-016 — rejected calibration is not a resize instruction:** inconsistent held-out
extent/depth evidence can reject a size hypothesis even when its fitting views
look plausible. Calibration now records the exact conflicting frame IDs and
per-frame relative disagreement. Rejection keeps the mesh; the runner and review
text distinguish resolving bad evidence from the separate accepted-hypothesis
resize/retrack path. The numeric limits and the mandatory preflight are unchanged.
Thin, nearly edge-on objects require particular care: a foreground mask can
contain mostly background-like depth while the inferred floor remains stable.
Inspect the native RGB/mask/depth together rather than deleting the held-out
failure or treating its proposed scale as an approved correction.

**R-017 — reject contradictory depth without inventing a surface:** a thin rigid
object turning edge-on can receive background-like predicted depth while a
smaller visible component remains consistent with reviewed views on either side.
The experimental `depth_evidence.qualify_depth` requires explicit bracketing
source evidence and substantial consistent foreground; it marks contradictory
visible object pixels missing, preserving occluders, other pixels and original
inputs. It never fills depths from an existing pose or a desired size, and does
not inspect calibration split labels. Insufficient support preserves the input.

A downstream bounded trial retained the original calibration sample IDs/split
and reduced held-out extent disagreement from 0.6771 to 0.1615 without changing
the 0.20 bound. However, a proposed smaller mesh still failed one of twelve native
seed-view gates after bounded registration refinement. Keep depth qualification,
size evidence and pose acceptance as separate outcomes; the depth result alone
is not a validated reference. No default filter, resize, gate threshold or actor
approval follows from this experimental helper.

**R-018 — RGB proposals need independent geometry checks and state rollback:**
qualified depth does not automatically recover object orientation through an
edge-on view. The reusable experimental `tracking_recovery` process-local adapter
uses the existing RGB refiner, checks each proposal independently against native
image, depth, declared support and motion limits, and tries bounded local/global
recovery before stopping. It restores the selected state before the next frame;
a failed tracker update cannot become the next accepted prior. JSONL evidence
records both actual selected states and rejected attempts. The ordinary backend
and its historical source snapshots remain unchanged. Floor support is not
silently treated as tabletop support. Full GPU acceptance is a separate,
thermally supervised downstream trial.

**R-019 — bind the actual exporter before a guarded run:** a container working
directory can outrank a supplied PYTHONPATH and expose an older module with a
compatible import name but an incompatible callable. `tracking_recovery.load_exporter`
prepends the selected packaged engine paths, verifies the exporter's exact source
file, and binds the full observed-seed call signature before model execution.
Preflight now calls that loader rather than merely importing a same-named module.
An initial downstream startup stopped before tracking on this mismatch; its
immutable attempt and source snapshot remain retained. The corrected trial uses
a separately pinned source snapshot and the existing bounded clock lease.

**R-020 — quantized proposals must fit inside their own motion bound:**
recording exact rejected poses exposed local recovery hypotheses whose intended
8 cm displacement became slightly larger after float32 quantization (for example
0.08000009 m). These candidates correctly failed the unchanged 0.08 m acceptance
check. Local proposal generation now reserves a coordinate-dependent float32
rounding margin before capping its correction. It does not relax the checker,
clamp arbitrary tracker outputs into acceptance, or bypass image/depth/support
checks. Five hundred sampled camera-frame configurations exercise the actual
float32 boundary; full tracking remains independently gated. Recovery logs now
include exact proposed/local poses and per-candidate rejection measurements.

A guarded full-clip retry after this fix still stopped on an image/depth/motion
conflict: silhouette IoU 0.8846 and depth error 2.33 cm required an 11.35 cm step,
while all local candidates inside the unchanged 8 cm bound failed depth. The
fix moved the stopping frame only two frames later. Both global registration
modes had zero temporally eligible candidates. No complete reference was
published. This demonstrates that a numerical defect can coexist with a deeper
tracking problem; fixing it does not authorize weaker checks or actor training.
Joint temporal registration is a separate research direction requiring a new
bounded protocol and independent held-out evidence, not an automatic fallback.

## Local pose repairs must preserve supported components and check their seams

When rotation is already image-supported, joint rotation/translation smoothing
can unnecessarily change good orientations, including frames between optimizer
knots. The reconstruction-repair module now offers an explicit translation-only
window mode that keeps every input rotation byte-for-byte while retaining the
same image, depth, support and displacement acceptance gates. It is opt-in;
existing joint fitting behavior is unchanged.

Independent local acceptance does not validate an assembled full trajectory.
The reusable local-window joiner requires unchanged endpoint anchors, source
frame identities, nonoverlap, and complete-clip continuity. Rejected or partial
hypotheses never silently replace their input. These are tested assembly and
optimization contracts, not a claim that any particular scene has been repaired
or that a robot can execute its motion. Campaign results live downstream.

A multi-frame proposal can reduce jitter while worsening previously good images
or human/object proximity. Rotation preservation alone does not prevent this.
The local-window API therefore supports conservative translation backtracking,
explicit editable-frame support, and an interaction veto after geometric checks.
Every smaller trial must pass the original thresholds; a smoother rejected
proposal remains a diagnostic rather than a replacement reference.

A predicted foot joint is above the actual sole surface; it should not be snapped
to a floor plane as if it were a contact point. Check the decoded human surface
against separately reviewed own-depth floor evidence before diagnosing root
height. Even a confirmed surface/floor gap does not identify whether the human
or predicted floor is wrong. Correcting the human alone must also preserve RGB
keypoints, silhouette, hand/object relationships and temporal motion; a root
translation must not bypass those guards.

## Validate the initial object pose before guarding subsequent tracking

A sequence can start from a model-ranked or previously recorded pose that has a
reasonable image outline but fails independent depth or floor nonpenetration.
Guarding only later updates cannot make that initialization valid. The reusable
`reconstruction.initial_registration` probe inspects at most 32 fresh RGB-ranked
frame-zero candidates and applies the same image/depth/camera/support criteria
used for recovery. It creates no seed if none passes. Frame zero has no invented
previous pose or temporal-step anchor; subsequent tracking keeps its usual step
bounds. No whole-clip acceptance follows from a passing seed.

Whole-clip mask IoU and a first-frame candidate score have different scopes.
Never promote a partial/failed tracking trial or relabel its seed score as a
complete reference improvement. The review renderer can explicitly show an
available prefix and blank later candidate panels with an unavailable-pose
notice; it does not carry or interpolate a rejected pose across missing frames.


## Review every annotated prompt frame against its own RGB

A wrong participant box may produce dense, nonempty background masks and fail
only much later during human-depth alignment. Inspect multi-frame prompt previews
and propagated human/object masks before model stages. Correcting subject identity
requires rerunning human inference and dependent metric geometry; verified raw
RGB-only depth can be reused with explicit provenance. Keep correspondence and
geometry acceptance limits unchanged rather than compensating for a wrong mask.

Source-box identity must be reviewed before geometry inference. A syntactically
valid box below the intended object can segment a coherent floor patch, produce
a thin metric mesh, and yield a complete pose timeline for the wrong surface.
Neither model completion nor IoU against that same predicted mask validates the
subject. Use `reconstruction.prompt_review` on the actual native source frame,
then inspect `reconstruction.mask_review` on propagated masks before metric mesh
inference. When only the object selection was wrong, reuse hash-verified human
and aligned-depth predictions; regenerate the object masks, metric mesh and track.
Compare old/new geometry against the same corrected masks, and keep camera,
source timeline and human estimates explicit. Semantic selections belong in the
caller’s configuration, not episode-specific branches in this public module.

Subject review is enforced by default before dependent human export, metric mesh
inference and object tracking. Acceptance binds the full mask catalogue, prompts,
source video and sampled review. A changed unsampled mask invalidates approval.
Disappearing/shrinking masks are review hints because occlusion is legitimate;
they cannot establish object identity or prove complete coverage. Explicit
unreviewed exploratory runs retain a bypass record. Existing frozen experiments
keep their original code and producing receipts; this default affects new source
snapshots rather than silently changing an active model job.

A segmentation mask can recover after occlusion while a 3D tracker stays lost.
Diagnose mask coverage separately from mesh silhouette agreement. Preserve a
good mesh/mask estimate and try a new RGB registration when the object is visible.
If using a local tracking window, slice every modality consistently and retain
source/local frame mappings, including for reverse-time checks. Do not claim a
correct hidden trajectory or silently splice a fresh pose across an unverified
join. Registration acceptance, window tracking, full-reference approval and
physics validation remain separate decisions.

Basic tracking execution can return a rigid pose and mark it producer-reliable
while its measured silhouette has zero overlap. Keep that raw producer flag for
provenance, but explicitly report conflicts with independent image evidence.
It must not suppress reconstruction warnings or stand in for depth/contact checks.
Fresh-registration refinement has no previous accepted temporal state: fit only
the visible frame, retain bounded hypotheses and require GPU revalidation before
tracking. CPU fitting success is not an occlusion or full-reference approval.

For a lost object track, independently registered poses on both sides of an
occlusion constrain a short recovery interval more directly than one-sided
propagation. `anchored_recovery` preserves both endpoint poses exactly and treats
the interpolated initialization as an unaccepted hypothesis. Intermediate
image/depth/support/motion checks still decide acceptance. Explicit RGB-reviewed
static occluder polygons can supplement person masks in declared intervals;
they must never suppress visible object pixels or be inferred merely because
a pose failed. Passing a short interval does not certify metric size or an
assembled full trajectory.

Empty object masks inside an explicitly bracketed recovery interval must be
declared before fitting and excluded from observed-image success. The anchored
wrapper now forwards explicit missing-image/depth IDs to the existing temporal
validator; observed evidence cannot be hidden by that mechanism. Inferred poses
remain labeled unverified. Audit native foreground depth before repeated pose
optimization: isolated multi-metre jumps under a small moving object can make
the depth objective incompatible with image evidence, even when the static
floor appears stable. Verify source/window byte correspondence before blaming
export alignment. Also retain multiple single-frame registration hypotheses:
independent maximum-IoU orientation choices can be temporally incompatible for
nearly symmetric objects. Neither a smaller rotation nor a good endpoint fit
alone validates the intervening trajectory.

Small-object depth can be checked with a camera-centered context crop and a
fresh full-frame control. Preserve focal geometry; an arbitrary off-center crop
does not satisfy a depth model that assumes a centered principal point. The
`foreground_depth` probe calibrates scale on reviewed static floor pixels outside
object/person masks and checks disjoint floor pixels. It never uses a candidate
object pose to manufacture matching depth. Two predictions from the same model
are cross-scale evidence, not independent truth. Missing object-mask frames stay
explicitly unobserved; a successful floor fit does not validate their motion.

`anchor_pair_review` ranks already-admissible endpoint hypotheses using only
predeclared intermediate fit images, then reports held-out agreement separately.
It cannot approve depth, metric geometry or a complete reference. Keeping several
orientation hypotheses is preferable to silently assuming independent maximum
single-frame silhouette fits form a consistent motion sequence.

Successful floor calibration can coexist with severe small-foreground depth
errors. In a bounded cross-scale probe, a closer view removed several large
foreground depth spikes but introduced another; a full-frame rerun repeated the
original failures. Retain per-frame regressions and avoid an automatic minimum-
depth selector or treating floor residuals as object accuracy. Visibility flags,
depth-proposal qualification and complete trajectory acceptance are separate.

Improving the first failed transition is not enough: a closer-view prediction
can remove that discontinuity and fail on the immediately following native
frame. `foreground_depth.audit_probe` reports floor passes, adjacent native
foreground-depth jumps, missing observations and cross-scale disagreements
separately. Its notice thresholds are diagnostic, not tracking acceptance
limits. Sparse sampled frames never establish continuity inside their gaps,
and the audit never selects a preferred per-frame depth or approves a repair.

A recent accepted tracking endpoint may narrowly satisfy its own depth check
while leaving no admissible next step. A bounded two-frame look-back can test a
small endpoint revision against its own observation, the previous accepted seam
and the next observation. Preserve earlier poses, all native gates and GPU
revalidation; do not substitute unconstrained smoothing or carry a stale pose.
The reusable `lookback_recovery.propose_pair` exposes this as a CPU proposal.

Seed provenance tags are a schema contract. Keep the canonical source enum and
put recovery method, native frame and evidence hashes in separate fields.
Validate that contract before loading expensive models. Classify startup/runtime
errors separately from scientific quality rejection so a software error cannot
be reported as failed tracking evidence or prematurely close a borrowed lease.

A two-frame CPU recovery returns two distinct hypotheses. Ordinary tracking
from its revised anchor may still fail to rediscover the qualified second pose.
Keep that second proposal available for explicit native GPU revalidation and
check its join to the GPU-verified anchor. Resume from the second pose only if
both observations and joins pass. Recovery receipts now declare both required
GPU source frames rather than leaving the continuation boundary implicit.

Window length includes its seed frame; a recovery budget counts subsequent
updates. Declare both counts in tracking input receipts and validate the update
budget before GPU startup. A passed registration probe can be reused after a
software-only launch failure when its proposal and source hashes still match;
the failed launch remains separate evidence from tracking quality.

Stable static-floor scale does not rule out frame-to-frame foreground depth
errors. A centered context crop can provide a useful alternative small-object
depth series even when a fresh full-image control repeats the discontinuity.
Keep calibrated full/crop controls, original depth, camera FOV and held-out
floor checks together. A few smoother samples motivate a bounded sequence
experiment with explicit boundary/pose checks; they do not approve automatic
depth replacement or continuity between those samples.

Lossless PNG does not imply an exact float-metres-to-integer-millimetres round
trip. A legacy truncating encoder can change an untouched region by one
millimetre even when the in-memory composite copied it exactly. Restore the
original decoded uint16 pixels outside the crop after encoding and verify the
whole native window. Keep the original archive immutable and reject writes to
a hardlink of it. This fix leaves the newly predicted crop values unchanged.

A temporal fit may substantially improve a difficult interval while slightly
regressing an already-admissible observation. An optional preservation proposal
can retain such original fit-frame poses, provided their depth/support/camera
checks also pass. Never preserve solely on silhouette score, use held-out poses
for restoration, or skip the subsequent full-window/motion/GPU checks. The
result can change acceleration; report this separately from acceptance and
physical followability rather than lowering an image-preservation gate.

Improved object tracking can change derived hand-contact phases even when the
retargeted robot root and every body joint target are identical. Validate the
new phases in physics rather than equating reference proximity with a grasp.
A measured opening prefix is a useful isolated reaching/contact diagnostic,
but cannot certify an unresolved pickup or placement tail. Keep each canonical
mesh/trajectory pair together in rendering, and separate prefix own-reference
scores from full-sequence competition scoring. See `PLACEMENT_TRACKING.md`.

Dense fit-frame pose initialization should retain previously qualified interior
anchors as well as endpoints. Dropping an interior anchor changes the comparison
hypothesis and can reintroduce displacement/rotation failures. Proposals remain
editable; the final comparison still uses the registered-anchor-only path.

A stronger depth objective can worsen both depth and silhouette agreement in
a coupled temporal fit. Retain the unchanged-objective control and bound such
retries; increasing an objective weight does not relax acceptance. Restoring an
otherwise admissible fit-frame observation can also create a motion-step failure
at its neighbours, so whole-window checks remain necessary after preservation.
Distinguish absolute native GPU validity from image-preservation/benefit gates:
passing the former does not remove a recorded comparison rejection or establish
a continuous reconstructed reference. Keep improved partial intervals separately.

Accepting an image-comparison warning does not make a direct interval splice
valid. A retained old prefix and a depth-revised interval can each pass their own
checks yet introduce a large camera-space jump at their boundary. Compose a
complete native-ID ledger, check every join, and revalidate any connecting tail
as one sequence. Preserve each interval's original gate result and its declared
depth basis; a prior measured prefix does not certify an unused regenerated
depth variant. Keep parent quality separate when packaging the new paired
human/mesh/pose artifact, then retarget and record fresh unassisted physics.

An improved reconstructed reference and higher object-tracking scores do not
establish successful manipulation. A frozen controller can follow the arm
motion while leaving the object on its support. Inspect saved physical
hand–object contact counts and actual object motion alongside task-success
rates. Keep kinematic reference contact phases separate from measured contact;
they may disagree even when reference retargeting and native pose checks pass.

During hand occlusion, a carton mask can retain its recognizable printed face
while dropping a visible low-texture bottom or side face. Inspect native-pixel
RGB/object/human-mask crops around the first tracking failure, not just a
downscaled full-clip contact sheet. Re-prompt the visible missing face and add
negative hand/background points; do not fill a hidden region across the hand.
Explicit `mask_feature_checks` should cover both printed and unprinted visible
faces and excluded occluders. Check nearby unprompted frames before accepting
propagation. A better mask does not prove that an independent depth/orientation
failure is fixed; revalidate existing registration seeds and retrack against
unchanged geometry/depth limits before replacing a trajectory.

Small-object masks may benefit from a fixed native-RGB crop before inference.
`reconstruction.segmentation_crop` maps prompt coordinates and backprojects
binary masks exactly into the original image. Keep the original camera/depth
for tracking, record the crop and check all original object-mask bounds first.
More segmenter pixels do not guarantee a better mask. Review unprompted frames
and excluded human/background regions, and retain failed trials. Feature-check
regions must lie inside a clearly visible object face; a rectangle crossing a
slanted silhouette is not a valid estimate of how much object area is missing.

When a rotating supported object becomes edge-on, reject foreground depth that
conflicts with visible geometry even if background floor checks pass. An observed
base pixel and a reviewed support plane can propose a camera-frame translation;
`reconstruction_repair.support_observation.support_pixel_seed` records that
assumption explicitly and never certifies contact or tracking. A free plane fit
that conflicts with the floor normal is a diagnostic, not a validated table.
An independently registered later visible pose can seed forward and backward
tracking, but the directional traces need explicit source mappings and ascending
seam validation before composing a complete reference. Missing intervals must
remain uncertain rather than being silently filled.

Report support-ray conditioning: a near-parallel camera ray can amplify small
plane-offset errors into large optical-axis depth errors. The support-pixel
helper returns this derivative as a sensitivity, not a confidence interval.
Do not regard a table observation as independent metric certification when its
normal, offset or visible contact point is ambiguous. Likewise, a small recovered
tracking island and one marginal CPU-valid boundary pose do not qualify a full
reference. Preserve failed multi-frame fits and the complete fallback instead
of admitting an interpolation that fails image or depth gates.

A free six-dimensional local refit can turn a useful RGB rotation into a poor
silhouette basin while reducing depth error. The opt-in translation-first
recovery holds the proposed rotation for a first fit, then falls back to the
ordinary bounded full fit. Initial-registration validity alone is insufficient:
recheck translation and rotation against the real preceding accepted pose.
The staged path remains a CPU proposal until independent GPU validation. Its
flag and finite per-call two-stage budget are explicit; default tracking and
shared acceptance limits do not change.

An isolated settling probe must use the packaged policy's initial object pose,
static support, solver/timing, collision decomposition, contact friction and
free-object damping. Testing only a ground plane or a different solver profile
can answer a different physical question. The policy-based probe records these
identities. Preserve failed displacement and late-drift measurements; target-GPU
confirmation is distinct from a CPU screen. A robot can transiently support an
unstable object, so completing a finite robot rollout does not override a failed
isolated gate. Pin measured collision hulls instead of silently decomposing the
same visual mesh again. Explicit training-scene matching is opt-in and records
whether any field actually changed; do not infer a mismatch merely because a
matching check was added.


Initial resting-contact diagnosis now includes exact imported collision screening,
source-preserving frame-zero-only reference revisions and explicit solver contact
buffer parity. A quieter object pose can still fail its image-preservation gate;
retain both outcomes. See [Initial rest admission](INITIAL_REST_ADMISSION.md).
# Native outline improvements and tracking-depth drift

A candidate can pass every native image/depth/support gate and stay under the
absolute per-frame motion limit while still adding many unnecessary angular
steps. Review the motion distribution against its parent before promotion.
The optional `outline_pose.review_motion` reports relative p95 roughness and
high-step counts; `select_observed_path` can constrain changes in pose
corrections relative to explicit original poses, preserving genuine source
turns. Fractional hypotheses at a single observed frame must each be measured;
they are not permission to interpolate through hidden temporal intervals.

Validate the actual RGB cache reader before GPU pose-estimator construction.
A complete HDF5 file can still omit a required JPEG metadata field. The public
`reconstruction.rgb_cache.write_frames` writes the complete JPEG100/4:4:4/
optimized contract and fails closed on timeline/layout changes. Native sequence
validation now checks reader metadata and endpoint observations before GPU
allocation. Tests exercise the vendored reader itself, including missing
`optimize`, rather than only checking file existence or duplicating the schema.

A better silhouette can still worsen depth or choose the wrong orientation.
The optional `reconstruction_repair.outline_pose` initializer preserves metric
mesh geometry, explicitly bounds pose changes and distinguishes depth-preserving
from depth-fitting experiments. Its convex outline proxy ignores holes and
concavities; validate the full source triangles, appearance, support and motion.

When image position and predicted object depth are stable but the estimated
camera-space pose briefly moves in depth, diagnose pose drift before rewriting
the depth cache. `depth_translation` can propose a bounded, image-centroid-
preserving camera-ray shift using the same frame's signed surface residual.
That is a hypothesis from predicted depth, not metric truth or hidden-motion
interpolation. `select_observed_path` chooses only supplied usable states at
every native frame and fails when no evidence-supported bounded path exists.
Do not fill that failure with an interpolated pose or relabel unchanged earlier
warnings as repaired. Application-specific results belong in downstream docs.

Sparse warning counts do not establish full-clip coverage. Before retargeting
a repaired suffix, inspect every unchanged prefix frame with the same native
image, first-surface depth and support checks. A depth-only discrepancy can
occur even when silhouette overlap is good. Small translation hypotheses must
also preserve appearance and pass whole-sequence velocity checks; image-centre
preservation alone is insufficient. Keep already verified intervals fixed and
reuse a human conversion only after proving the mesh-local and world-frame
transforms across the complete original sequence. GPU source validation,
policy-matched isolated settling and frozen physics remain separate receipts.

Collision replay must resolve omitted optional solver fields through the same
`SimConfig` defaults used during execution. An otherwise valid policy may omit
`enable_multiccd`; requiring that YAML key breaks replay without indicating a
physics failure. Keep diagnostic-only fixes separate from immutable execution
snapshots. Implausible mesh contact witnesses should also be checked against
the actual solver's saved-state contacts and explicit placement geometry before
changing a stable support scene; a narrow-phase witness is not a global mesh
intersection measurement.

A CPU-only preflight embedded in a GPU worker must run in a separate CPU
subprocess, with CUDA hidden before imports. Passing a standalone CPU test does
not cover the later admission path from a GPU-enabled queue. Do not hide CUDA
in an already initialized training process or weaken the preflight's device
check. Test both contexts and verify that the parent retains its GPU settings.

Floor disagreement can involve mesh shape, pose and predicted depth at the same
time. A bounded underside correction may improve the first view while later
views still fail support checks. Keep whole-sequence approval separate from
that partial benefit. Camera-ray projection can require large edits at oblique
viewing angles; the opt-in support-normal proposal minimizes displacement but
must report that its silhouette changes. Neither mechanism proves hidden
geometry or permits relaxing source acceptance. Independent single-frame refits
also require native validation and temporal seams before use as a trajectory.

Inspect controller failure before changing finger closure: a robot can lose
balance before any intended contact phase begins. Report root/palm drift before
that event and compare matched unassisted trials; additional finger reward
cannot establish that an unreachable or missed approach has been repaired.

Before regenerating a full depth sequence, compare fresh native-frame RGB
probes with a retained floor-stabilized depth candidate. Agreement can justify
hash-bound reuse as an experimental input, never reuse of old pose acceptance.
Keep the source frame mapping, masks and camera calibration explicit. Same-model
full-image/crop agreement remains prediction consistency, not independent metric
truth. Validate tracking again after changing either depth or mesh geometry.

Select a vendored engine before importing its legacy top-level packages in a
preflight process. Importing an image-owned package first can pin the wrong
package in the module cache even after search paths change. Keep source-origin
checks strict and test the actual reader in the same import order as execution.

A quality rejection after a finite CPU-recovery allowance is used is not proof
that CPU recovery was attempted at the failed frame. Failure events now record
used/maximum calls, attempts on that frame and remaining allowance separately
from the geometric rejection. Preserve the accepted prefix and unchanged quality
limits; any bounded continuation needs explicit input/frame correspondence.
The diagnostic does not authorize more work or promise that more search helps.

Per-frame acceptance can still produce a staircase trajectory: many repeated
orientations separated by occasional large accepted turns. Its95th-percentile
rotation step can even be lower than a smooth reference if jumps affect fewer
than5% of transitions. Count large steps as well as quantiles, verify source
appearance and temporal seams, and use the existing motion review before actor
admission. A new regression test covers this sparse-jump failure mode.

A floor-supported pose model can suppress motion jumps yet fail source views
or predicted foreground depth. Fixing tilt is especially restrictive for an
object being tipped; optional bounded tilt still does not establish the correct
contact geometry. Keep these as explicit, independently measured hypotheses.
Whole-clip tracking coverage, plausible floor contact, image agreement and
temporal coherence are separate conditions. Prefer a recorded rejection to
silently increasing a support/depth tolerance or promoting a discontinuous
sequence merely because its tracker reached the final frame.

Temporal smoothing should initialize observation fitting, not replace it.
Regularizing camera-space object centres avoids apparent translation caused
only by changing a mesh-local rotation pivot. Keep anchor poses exact and bound
rotation corrections in a local SO(3) chart. Sparse joint contour/depth/support
fitting can make whole-clip CPU proposals practical, but its proxy cost can
improve while exact triangle overlap or floor contact regresses. Independently
check every native view, retain rejected revisions, and select only measured
per-frame alternatives through explicit temporal constraints. Small translation
polishing must use the declared silhouette visibility convention and, when
supplied, reject support penetration; a visible-mask pass is distinct from a
full-mask pass.

Distinguish a floor-only simulator scene from a missing tabletop asset. A
tabletop placement audit cannot run without explicit support triangles; do not
silently treat that absence as successful placement. For a deliberately
floor-only scene, an opt-in audit can instead take the simulator's explicit
unit world plane and test all collision vertices against it. Preserve the
declared maximum gap/penetration and report that clearance alone is neither
settling nor real-world calibration. Keep diagnostic audit revisions separate
from the frozen numerical policy build when reusing already prepared policies.

Preserve both dense triangle-surface and sparse native vertex-depth checks at
handoff boundaries. They use different pixel cohorts and can disagree by a few
millimetres near the same numerical limit. Reproduce the native sample set,
depth dtype and float32 pose boundary before assuming the renderer or depth is
wrong. Qualify a small fixed-rotation translation against both measurements,
visibility, support and both neighbours, then validate again on the GPU without
relaxing limits. Keep the accepted prefix and rejected receipts immutable.
Translation-only helpers should not round the rotation matrix when simulating
the native float32 boundary; cast a temporary pose for measurement instead.

Diagnose physical root translation and heading separately from reconstructed
object accuracy. A late source-pose improvement cannot explain body drift that
already occurs in an identical early prefix. Compare actual and kinematic root,
heading and palm errors at explicit timestamps, and report any pre/post-step
sampling offset. Verify the pretrained model's upstream observation convention
before labelling missing global feedback as a wiring bug: SONIC's future
orientation tokens are reference-relative, not actual-to-reference heading
errors. An added heading loop is therefore a new controller experiment. Keep
it opt-in, bounded and auditable, retain an unchanged control, and avoid hiding
failures with hand tuning before reliable approach is demonstrated.

A downstream three-repeat ablation rejected bounded future-yaw conditioning
and its combination with small leg-reference target corrections as approach
fixes. The latter caused substantially earlier balance loss. A larger AUC can
coexist with worse approach and fewer contacts when the object remains near its
initial pose. Report body errors, upright intended contacts, object displacement
and success rates alongside score changes. Keep these controls disabled by
default, preserve rejected evidence, and inspect source-supported foot motion
and walk-to-stop timing before increasing gains. Near-floor foot-origin speed
alone is only a diagnostic hint; it does not prove a planted foot is sliding.
# Finite depth corrections need independent acceptance

Small signed surface-depth residuals can coexist with large median absolute
depth discrepancy when residual signs are mixed. A three-proposal residual
heuristic can therefore stop short of a feasible translation. The optional
`depth_translation.finite_ray_proposals` and `depth_guided_update.finite_ray_search`
provide an explicitly bounded alternative without changing the default tracker.
They preserve rotation, scale and centroid projection; measure correction from
the failed proposal and temporal motion separately from the last accepted pose.
All image, absolute-depth and support gates remain authoritative.

Predeclare the finite grid and candidate-ranking rule. An interior candidate
with margin to depth, correction and temporal bounds may be preferable to the
lowest-depth-error candidate at the edge of the permitted range. Preserve the
entire search evidence and independently validate the selected pose with the
native renderer. CPU acceptance alone must not update tracking state. Even a
native-valid isolated correction can fail on the next ordinary tracker update;
stop at that failure and retain the accepted prefix rather than filling a gap.
