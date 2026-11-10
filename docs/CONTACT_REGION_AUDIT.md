# Contact evidence before paired grasp planning

`egocentric.contact_observations` projects object-local contact points and normals
through calibrated camera poses, tests visibility, and measures optical-axis depth
residuals in metres. It separates missing evidence from disagreement. Occluder and
object masks exclude background/foreground pixels; the scene adapter additionally
rejects mesh self-occlusion. Observation support requires enough visible frames
and both median and p90 error limits. Missing samples never become zero errors.

`retargeting.contact_proxy.local_span_audit` compares a local solid span in the
source mesh with the union of convex collision pieces. Internal decomposition
faces do not count as object boundaries. Its error limit is the smaller of an
absolute metric bound and a fraction of wall thickness. This is one local line
check, not certification of the complete contact patch or object topology.

`controller_actor_training.contact_region_audit` runs these checks against a pinned
plan and reconstruction. It verifies scene assets, source object/pose provenance,
and the settled plan before sampling. It converts task-world contacts back to
object-local coordinates, then uses the original reconstructed object/camera poses
for each capture frame. A gravity-settled task pose is never substituted for the
observed motion. Inputs and worker sources are hashed. Outputs include per-contact,
per-frame evidence, capture timestamps and annotated RGB images.

The CLI accepts a downstream JSON configuration:

```bash
python -m real2sim2real.controller_actor_training.contact_region_audit --config audit.json
```

Required fields are `source_config`, `source_config_sha256`, `plan_sha256`,
`reconstruction`, a fresh `output`, distinct `candidate_ids` (at most 16), distinct
capture indices `frames` (at most 40), and `deadline_s`. `image_sampling` specifies
`radius_pixels`, `minimum_depth_pixels`, `minimum_facing_cosine`;
`observation_limits` specifies `minimum_frames`, `max_median_error_m`,
`max_p90_error_m`; `proxy_limits` specifies `max_boundary_error_m` and
`max_relative_error`. `self_occlusion_tolerance_m` is explicit. Object mask ID 1
and occluder IDs 2,3 are overridable. This scene adapter needs the optional geometry
and MuJoCo runtime but runs on CPU. The standalone projection utility does not.

Reused reconstruction depth is a consistency check, not independent ground truth.
A thin wall may occupy fewer pixels than the sampling window. Preserve the primary
audit, then run smaller-window sensitivity checks without fitting poses or changing
error limits. A point passing in one window does not verify an entire grasp.
Residuals may reflect geometry, pose, calibration, stereo matching or mixed pixels;
this audit does not identify which component is wrong.

## Paired contact arrangements

`retargeting.paired_contacts.paired_wrench_screen` combines two contact sets in the
same metric world, retaining a distinct force budget for each pad. Samples on a
single pad share that budget. Missing nearby opposed samples reject the arrangement.
The existing conservative static friction-pyramid solver tests gravity force and
torque balance; it does not establish dynamic success or actuator feasibility.

`controller_actor_training.paired_contact_diagnostic` enumerates opposite-side
candidate centers around the shared COM. It verifies pinned plan/audit hashes and
matching mass, friction and force assumptions. It reports static balance separately
from observation support. `ready_for_robot_pair_fitting` requires both. Even that
flag does not mean control readiness: the two inputs are separate single-hand
hypotheses, not a jointly fitted robot pair. No original candidate is rewritten.

```bash
python -m real2sim2real.controller_actor_training.paired_contact_diagnostic --config paired.json
python -m real2sim2real.controller_actor_training.contact_region_review \
  --audit primary/audit.json --audit sensitivity/audit.json \
  --paired paired/diagnostic.json --output primary/review.html
```

The paired configuration specifies `plan`, `plan_sha256`, `audit`, `audit_sha256`,
`candidate_ids`, a fresh `output`, and `deadline_s`. Reports are static HTML/images;
serve the output directory on localhost for browser viewing. No control video is
implied. The review labels observed depth separately from agreement.

Before a supported pair can advance, fit the actual two robot hands jointly,
check hand-hand/object/table collisions along the approach, verify joint/actuator
limits, then run unassisted lift/hold and perturbation trials under the original
acceptance criteria. Static balance alone must not unlock policy training.

Tests cover metric projection, occlusion/missing-depth rejection, local proxy
unions, thickness-relative error, separate pad budgets, missing patch rejection,
input provenance and the separation of static balance from promotion.
