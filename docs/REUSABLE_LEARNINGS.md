# Reusable reconstruction and grasp lessons

The implementation keeps capture-specific data and outcomes downstream. This
index maps findings to reusable mechanisms, checks and explicit remaining work.
It is not an assertion that a new scene, hand or robot has been validated.

| Finding | Reusable implementation | Verification / interpretation |
| --- | --- | --- |
| A moving camera cannot define the simulation world | `egocentric` calibration, camera motion and upright support; `object_task` metric scene export | Explicit units, source hashes and world transforms; finite observed table extent remains an assumption |
| Crop localization and pose-model improvements must be compared separately | `hand_experiment`, `hand_tracks`, shared-crop HaMeR/HaWoR workers | `test_video_hand_observations.py`; common fitter and matched observations |
| Appearance tracks do not guarantee finger identity | `hand_point_tracks`, `hand_observation_policy`, `multiview_points`, `hand_anchor_audit` | Point-track and observation-policy tests; ambiguity/unknown evidence remains explicit |
| Wrist depth crops can include watch/forearm outside MANO | `hand_depth_support` prepares image-defined support before fitting; `hand_support_sensitivity` audits margins | `test_hand_depth_support.py`; preserve fit/reserved sample separation and missing boundaries |
| Smoothing or fallback coverage cannot certify corrected motion | `hand_sequence`, `point_hand_sequence`, `interaction_readiness` | Sequence and readiness tests preserve rejected components and distinguish reference admission from control validation |
| Human fingers can remain uncertain while robot task planning proceeds | `object_task`, `retargeting.object_grasp`, `grasp_kinematics` | Width/height/joint/contact checks are mandatory; optional cue influence is bounded |
| Collision simplification can remove usable geometry | `object_grasp_scene` checks closure and sampled convex-proxy errors, retaining source visual geometry and inertia | An approximation audit is not a topology certificate; robot/collision assets remain external |
| A reconstructed pose may not be a stable physical start | Explicit reconstruction-preserving or gravity-settled task initialization | Displacement/rest checks retain the rejected pose and label a new task start; no silent reference repair |
| A geometric contact plan is not a successful grasp | Separate contact IK, approach search, free-object simulation and hold evaluation | `test_object_grasp.py`; reject object actuators, welds, gravity assistance and stale inputs |
| Five target fingertips do not imply five loaded fingers | `grasp_hold.contact_support` reports actual per-group normal force and active sample fraction, including palm/other geometry | `test_grasp_hold.py`; contact evidence is not a force-closure proof |
| Wrist rotation can be confused with slip | `grasp_feedback.relative_translation_drift` uses the moving wrist frame; `object_grasp_audit` rescores retained states | Rigid-rotation invariance tests; original world-vector metric and historical verdicts remain available |
| Brief lift success can hide later drift or changed support | `grasp_hold.measure_hold` / `hold_rejections` inspect all retained hold samples; `grasp_robustness` separates development and longer frozen checks | Later-sample failure, nonfinite evidence, missing hold and independent contact/penetration gates have regression tests |
| More closure can cause penetration instead of improving grasp quality | Bounded `contact_closure` and `orientation_correction` under unchanged joint/force limits | Feedback tests verify direction, speed/travel bounds and singularity behavior; failed trials remain failed |
| A diagnostic follow-up cannot erase development failure | Pinned comparison/protocol selection in `grasp_robustness.validate_fixed` | `validation_decision` tests prevent later passes from silently promoting a rejected controller |
| An FFmpeg decode is not a playback check in an IDE | `review_media` supplies embedded primary WebM and an image fallback; `review_server` supplies loopback HTTP with byte ranges | Conversion/cache/HTML tests, range/path tests, plus actual browser playback and seeking checks |

## Applying the modules to another case

Keep a new capture's calibration, assets, configs and experiment results outside
the public package. Prepare a metric scene with explicit frame conventions and
finite support. Review collision approximation and settling before robot contact
search. For a different hand, review joint/site ordering, geometry-to-finger names,
posture convention, actuator limits and feasible grasp span. The current wrist
fixture does not establish an arm's reachability.

Use the same observations and fitter when comparing localization/pose models.
Keep unknown hand identities unknown; reserve evaluation observations before
fitting. In object-centric mode, declare the object task independently of human
articulation. Evaluate actual simulated support, minimum object clearance,
wrist-relative drift, object orientation and penetration. Freeze selection before
longer or altered-parameter trials, and retain the original outcomes.

Report the extent of validation: observed sample intervals, nominal material
assumptions, fixed-grasp parameter cases, and the sensing available to feedback.
CPU tests, deterministic repeats and saved-state renders serve different purposes;
none alone establish independent reconstruction accuracy or hardware success.

## Findings that require new work

The need for new evidence or a new robot capability is recorded explicitly rather
than represented as implemented functionality. Independently identified human
joint/contact observations still need acquisition for faithful imitation. New
bimanual contact planning, arm integration and a task-specific learning environment
are not implemented by the wrist-fixture pilot. Exact simulator pose/contact
feedback remains privileged until a deployable observation path is supplied.
The existing training packages do not automatically make this new task ready for
policy training.

See [object-centric grasp workflow](OBJECT_CENTRIC_GRASPS.md) and
[test instructions](TESTING.md) for interfaces and validation commands.
