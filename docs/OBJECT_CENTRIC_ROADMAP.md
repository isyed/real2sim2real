# Task-first control, then reference fidelity

Keep reconstruction and physical task control as separate, testable capabilities.
A robust object-centric controller can supply a warm start for demonstration
tracking; task success by itself does not establish imitation accuracy.

## How the available projects fit

This project owns calibrated reconstruction, metric scene artifacts, uncertainty,
explicit source hashes, contact planning and reproducible evaluation. The
`do-as-i-do` project supplies useful reconstruction and demonstration-retargeting
components; a complete demonstration-driven pipeline still depends on adequate
hand/object observations. A lack of reliable finger evidence is not resolved by
changing the orchestrator.

[RoboSimStudio](https://github.com/robosimstudio/robosimstudio) is a promising
simulation and task-execution backend. Its
[GaP integration](https://github.com/robosimstudio/robosimstudio/blob/main/pyproject.toml)
connects graph-based robot tools with separately discovered skill bundles.
The [skill inventory](https://github.com/graph-robots/open-robot-skills) includes
grasp proposals, reachable-grasp selection, grasp verification, held-object
registration, carry planning and execution. These can organize task phases and
recovery without first training an entire task policy.

The inspected robot-tool interface uses arm TCP goals and gripper descriptions.
That is useful for arm integration, but is not evidence of an already validated
two-dexterous-hand contact controller for an arbitrary object. A skill's exit
condition must be backed by suitable observations and physical evidence.
Changing simulator/solver/contact representation also requires a new matched
baseline; success cannot be transferred by copying a scene's appearance.

Recommended architecture: retain the metric reconstruction/evaluation interfaces,
make any RoboSimStudio/GaP backend optional and independently installed, and bind
validated grasp/carry/verify primitives to graph nodes. Do not import a sibling
checkout at runtime. Pin backend and skill revisions, object collision/inertial
assets, frame conventions, robot limits and observation sources. Verify scene
equivalence and free-object contact before comparing outcomes across backends.

## A success-first curriculum

1. Establish a stable grasp/lift/hold with fixed gates and matched parameter
   cases. This is a fixture-level subtask, not complete episode success.
2. Add actual arms, reachability and collision checks, then complete the intended
   object task with explicit outcomes and recovery. Evaluate repeatability under
   plausible perturbations.
3. Preserve that controller as a baseline and introduce trusted reconstructed
   object waypoints, orientation and timing as progressively stronger objectives.
   Keep physics/contact/task constraints active. Use uncertain human wrists and
   fingers as confidence-weighted preferences, with missing evidence masked.
4. Compare the success/fidelity tradeoff on frozen evaluation cases. A reference
   objective that loses the grasp is not an improvement. Distillation or residual
   policy training can follow a reproducible control/observation/reset interface.

The useful reference for object-tracking metrics is primarily the **object pose
trajectory**. Matching every human finger is neither sufficient nor generally
necessary for matching object motion with a different robot embodiment. A good
grasp may need replanning when later object orientations or arm constraints differ.

## Metric discipline

The challenge links AUC/MPPE to object tracking and uses a separate end-to-end
evaluation entry point for the egocentric track; see the
[challenge specification](https://nvidia-isaac.github.io/video_to_data/v2d_challenge/).
Check the exact evaluator/version for a particular benchmark before reporting
officially comparable values.

The included `flash_chord.evaluation.metrics` computes ADD AUC from object-surface
pose errors across distance thresholds (higher is better), and MPPE from six
body-local keypoints transformed by achieved/reference object poses (lower is
better). These are not finger-articulation error measures. A pot can be lifted
successfully along a different path and still score poorly against a timed
demonstration. Report physical task success and tracking metrics separately.

Use the same metric world, object identity/geometry, initial state and sample
times required by the evaluator. Arbitrary time warping, reference changes or
post-hoc alignment cannot silently improve the score. An explicitly settled
task start is a different experiment from preserving the demonstration's start.
Observed reconstruction is available to control; held-out clean trajectories
remain evaluation-only. Improving reconstruction accuracy and retargeting its
supported motion are separate from modifying a reference to make control pass.
