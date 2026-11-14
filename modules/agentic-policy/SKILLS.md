# Bundled skill sources and continuous-simulator adaptations

All **33** higher-level skills from open-robot-skills revision
`752dfbd96968d4d08379f8111765efbf6457800b` are included with their supporting
Python scripts, workflow JSON and references. The 145 source/license files are
byte-identical to that commit. `_vendor/robot_skills/manifest.json` records every
SHA-256 and skill's license declaration; its LICENSE retains upstream terms.
No neighboring checkout or network download is needed after installation.

These resources are **source references requiring adaptation**. They still
contain upstream imports and connector calls. In particular, opening a simple
gripper is not equivalent to coordinating an articulated hand's fingers, and
an external motion planner is not implicitly installed. Copying an upstream
script unchanged can therefore fail. The author can read all of them, but must
use the actual `tools.json` contract when producing executable policy code.

The author receives `skills.json` and the same catalogue under
`CODE_ACCESS.json.skills`. Each source entry identifies its definition, source
directory, license, missing declared tool calls and relevant project adaptations.
The missing-tool list follows upstream metadata; it is not exhaustive runtime
dependency analysis. No source entry is marked executable merely because a tool
name happens to match. Optional perception models, cable skills and external
planners remain references until an explicit integration is supplied.

## Executable adaptations

These are separate project implementations inspired by the corresponding skill
operations. They are registered only by the continuous physics backend and run
through the same isolated graph-to-host RPC as the original robot commands.

| Tool | Behavior | What it does not establish |
| --- | --- | --- |
| `robot.fit_grasp` | Fit wrist IK and supplied pad positions/normals in scratch state | Insertion clearance, contact force closure or loaded retention |
| `robot.acquire_grasp` | Screen and execute explicit opening/insertion/closure joint waypoints; enable bounded pressure afterward | A successful lift/hold merely from reaching the closure command |
| `robot.plan_transport` | Convert desired object poses to wrist poses using the measured object-to-wrist transform; seeded IK and sampled path checks | Global motion planning or guaranteed rigid attachment |
| `robot.execute_transport` | Execute a host-retained plan if state, commands and pressure targets match its planning snapshot | Retention under load merely from plan admission |
| `robot.verify_grasp` | Explicit timed hold; check sampled pad forces, absence of table support, and relative translation/rotation drift | Proof of opposing normals, force closure, future motion robustness or hardware validity |
| `robot.place_and_release` | Execute placement approach; check the frozen target/support conditions; invoke supported-dwell release | Completed task before final placement-window and home evaluation |
| `robot.verify_placement` | Check the recorded stable window against the same physical placement predicates as the independent evaluator | Same-home return, complete horizon or strict control acceptance |

`robot.describe_skills` reports the coordinate convention, diagnostic checks and
limitations. Source skills such as `grasping-direct-ik`, `grasping-with-planner`,
`planning-held-object-motion`, `executing-held-object-motion`,
`transporting-objects`, `placing-held-feature` and the verification skills link to
these adaptations. Their complete upstream APIs have not been emulated.

All poses are right-handed 4x4 world transforms with translation in metres.
Pad targets and inward normals are world-frame vectors. Joint commands are full
maps in native radians; force targets are newtons. The current backend owns one
arm/hand. Multi-arm coordination needs an explicit backend rather than choosing
an unimplemented side argument. Observations are privileged simulator state.

## Timing, motion and rejection

Planning/fitting/verification of saved placement records do not advance physics.
Only explicitly requested path, release and hold durations do. No adapter adds
a diagnostic pause. The existing rest-to-rest interpolation within each path
segment remains unchanged; waypoint velocity semantics belong to the policy.

Acquisition validates both path and pressure configuration before moving. It
does not invent an opening posture or insertion corridor: the proposal supplies
those joint waypoints. Pressure is enabled after the closure path and takes
effect on the next explicit movement/hold. Invalid commands fail before motion;
a dynamic failure leaves the simulated state at its actual failure point.

Transport screening checks both the existing robot path with the object at its
current pose and a predicted carried-object sweep. The latter uses the measured
relative transform only in scratch state; the live object remains free. It checks
contacts against fixed obstacle bodies as well as the world. Both screens must
pass. The original fixed-object screen can conservatively reject a path that the
carried-object prediction admits. Sampling is at the configured feedback period;
neither screen is a continuous collision certificate. Object slip and dynamics
still require simulation. Callers supply clearance-first lift, inversion and
placement waypoints; the adapter does not search alternative obstacle routes.

An admitted plan is kept in the host and addressed by its ID. Advancing physics
or changing commands/pressure makes it stale. The policy must replan rather than
executing an old grasp-relative transform. Tool calls do not reset or weld the
object or modify scene, gains, task timing or acceptance limits.

Release checks the frozen task location/orientation, support and speed before
calling the original dwell-gated release tool. If dwell is insufficient, it
returns a reason without silently waiting. Finger opening and any wrist
withdrawal still undergo the existing path and loaded collision checks.

## Diagnostic checks versus campaign acceptance

The grasp adapters expose their diagnostic defaults through `describe_skills`:
pad fit error at most 5 mm, normal error at most 25 degrees, at least two pads
above 0.05 N, support fraction at most 0.1, relative drift at most 10 mm/10 degrees,
and a requested hold of at least 0.1 s. A downstream host may explicitly set
`runtime.skill_checks` before freezing a campaign. These defaults are interface
diagnostics, not recommended universal control limits or changes to an existing
campaign's acceptance criteria. Counted contacts do not by themselves establish
opposed normal directions or grasp wrench feasibility.

The hold reference is measured after acquisition, with exact window endpoints
and intermediate logged observations. Gaps exceeding the configured logging
period reject coverage. This is sampled evidence; faster contact losses require
finer telemetry. The independent final evaluator still decides task completion,
home return and configured downstream admission. Graph status and diagnostic
skill success cannot overwrite those verdicts. Conditional geometry remains
conditional in every stage.

## Maintainer refresh and validation

```bash
python modules/agentic-policy/tools/vendor_robot_skills.py \
  --source /path/to/open-robot-skills \
  --revision 752dfbd96968d4d08379f8111765efbf6457800b
```

The refresh tool reads committed Git objects, not uncommitted checkout content.
Wheel and source-distribution builds include the resources. Clean-install tests
verify all source hashes and author readability without workspace mounts.
MuJoCo tests cover unchanged acquisition commands/timing, read-only planning,
carried-object obstacle rejection, stale plans, support-gated release and contact
loss diagnostics. An isolated graph smoke test exercises the actual portable
robot scene separately from an ASL proposal or task-success claim.

### Whole-environment support observation

`robot.observe_environment(surface_geom)` reports the whole carried object's
signed clearance from an upward surface reference, finite footprint overlap and
forces from all fixed nonrobot environment contacts. This is separate from a
runtime's configured `support_fraction`, which may intentionally include only a
placement rack and omit the table. Zero configured support is not proof that an
object is airborne. A commanded wrist rise is not measured object clearance.

The observer forwards a scratch copy of MuJoCo data and never advances live
physics. Forces are current-state scratch estimates, not cached forces from the
original last integration step. Box/plane surfaces use their top plane. Mesh
surfaces use a conservative gravity-aligned highest-vertex plane and projected
bounding intervals: positive separation bounds vertical clearance; negative
separation alone does not establish collision. These checks do not certify a
swept path or unobserved reconstruction. Use actual contact evidence and the
existing complete object/hand transport screens alongside them.
