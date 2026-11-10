# Search observed contact sites before fitting robot hands

`retargeting.contact_site_search` samples reconstructed face centers with
deterministic farthest-point coverage in position/normal space. Normal separation
keeps opposite sides of a thin sheet distinct. Sampling is bounded and does not
claim exhaustive surface coverage. It is separate from `retargeting.surface_support`,
which reconstructs furniture/support surfaces from depth.

`controller_actor_training.surface_support_search` runs a CPU survey against a
pinned calibrated capture, reconstruction, task scene and contact plan. It keeps
all geometry, poses and physical assumptions fixed. Model-based visibility,
target object/hand masks and direct stereo matching supply candidate evidence.
The matcher searches its full declared disparity range independently of predicted
depth, retaining missing/ambiguous observations as failures.

Discovery frames alone rank candidates. A frozen `shortlist.json` is written before
held-out observations are evaluated. Shortlisting requires the configured minimum
frame count and median/p90 depth consistency; nearby similarly oriented sites are
deduplicated. Opposite-facing sites remain distinct. An empty supported discovery
set stays empty rather than triggering a fallback grasp. Held-out failures remain
rejected, even if a metric narrowly misses its limit.

Two sampling modes are supported:

- `whole_surface`: distribute hypotheses across the complete reconstructed object.
- `upward_force`: restrict sampling to faces whose circular friction cone permits
  a positive upward force in the settled task world at the original friction.

For an outward unit normal `n` and coefficient `mu`, the maximum upward force per
unit compressive normal force is `-n_z + mu * sqrt(1 - n_z**2)`. Positive capacity
is necessary for a site to contribute lift, but does not establish horizontal or
torque balance, finite contact area, actuator feasibility or robot access. Normals
come from the reconstruction; depth agreement alone does not verify them.

`controller_actor_training.support_arrangement` combines sites passing both stages
of pinned surveys sharing the same scene, calibration and criteria. It deduplicates
identical source faces, then uses the existing conservative gravity-wrench solver
with unchanged mass, friction and per-site normal-force limits. It also reports an
optimistic upward-force upper bound, ignoring lateral balance and moments. A bound
below object weight rules out that sampled arrangement under those assumptions.
Nearby points are not automatically separate realizable robot pads: even a positive
result still requires finite-patch and robot-hand validation.

```bash
python -m real2sim2real.controller_actor_training.surface_support_search --config survey.json
python -m real2sim2real.controller_actor_training.support_arrangement --config combined.json
python -m real2sim2real.controller_actor_training.surface_support_review \
  --arrangement combined/arrangement.json --output review/review.html
```

Survey configuration fields are `capture`, `calibration`, `plan`, `scene_receipt`,
`reconstruction_artifact` and their `_sha256` values; `reconstruction`; a fresh
`output`; `deadline_s`; disjoint `discovery_frames` and `validation_frames` (at most
40 total); `sampling` (`count`, `normal_distance_m`); `sampling_mode`; `shortlist`
(`maximum`, `minimum_spacing_m`); `observation_limits` and `matching` as documented
in [Stereo contact evidence](STEREO_CONTACT_EVIDENCE.md). Optional `review_frames`
select overview images. Results include sampled faces, both evidence stages,
per-frame observations, initial table heights, force diagnostics and source hashes.

Combined configuration supplies a `surveys` list of `{path, sha256}` entries,
a fresh `output` and `deadline_s`. Neither CLI runs hand IK, physics or training.
The browser report displays static projections; cyan means supported point depth,
not a certified grasp. All inference data and scenario-specific configs remain
downstream.

Tests cover repeatable sampling, face eligibility, separation of discovery from
validation, opposite sheet faces, empty shortlists, upward-force signs, provenance,
deduplication and exclusion of held-out failures from force screening.
