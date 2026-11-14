# Freezing an external-author campaign

`agentic-policy` supports manual, reviewed and automatic proposals using the
same trusted simulator. `feedback_revisions: 6` means one initial proposal plus
six measured revisions, not six independent initial guesses. A from-scratch
condition has no `seed`; the private manual comparison remains a separate bundle.

For per-proposal human inspection, use `mode: review`: `advance` stops before
physics, then `submit` evaluates one candidate and returns. Inspect the resulting
video and contact evidence before the next `advance`. Automatic mode does not
pause for this review. Repeated clock resets warrant the
[simulation stability audit](SIMULATION_STABILITY.md) before further revisions.

Before author execution, freeze `experiment.json`, the portable scene, initial
states, explicit development cases, task gates and evaluation definitions.
`init` copies and hashes these inputs. `preview-author` prepares the actual
initial packet without calling a model or consuming a proposal. All changes
must be made before the campaign is authorized and started.

A user-requested budget extension can append revisions after the existing budget
finishes, without editing the frozen bundle or archived proposals:

```sh
r2s2r-agentic extend --campaign ./campaign --additional-revisions 4 \
  --reason 'User requested four further feedback revisions'
```

This records a budget amendment in campaign state; `total_proposals` becomes the
effective cap while the original `experiment.json` remains unchanged. It makes
the previous external-author authorization scope invalid, so record the user's
extension approval against the new scope before `advance`. No author is invoked
by `extend`. Keep final held-out results out of any subsequent feedback; extending
a campaign does not turn held-out evaluation into development data.

```sh
r2s2r-agentic init --bundle ./bundle --output ./campaign
r2s2r-agentic preview-author --campaign ./campaign --output ./packet-review
r2s2r-agentic status --campaign ./campaign
```

The author command is an explicit argv list, with `{workspace}` substitution,
bounded timeout, read-only mounts and optional pinned file hashes. The optional
`codex_author` adapter uses an independently installed Codex CLI and its existing
authenticated account. No provider binary, account or credential is bundled.
Its `--check-only` mode checks binary/file availability without a model call;
run that check with network disabled. Linux bubblewrap is the outer filesystem
sandbox. The adapter disables the CLI's inner sandbox only inside that outer
sandbox. Network, when enabled for the provider, is not a domain-filtered proxy.

Mount and hash the complete CLI distribution directory, including
`codex-code-mode-host` and its resources. Mounting only `codex` can pass a version
check but leave the author unable to use tools. The adapter checks both
executables before any model call. Archive infrastructure-only attempts separately
from actual policy proposals; a runtime repair must not silently change scene,
criteria or candidate budgets.

External authors require `author-authorization.json` beside the campaign state:
`approved: true`, `scope_sha256` from `authorization.scope(state)`, and a
`user_decision_reference` to the explicit approval. This is an audit interlock,
not a cryptographic permission system. The scope binds the frozen bundle, author
configuration and proposal budget. Never create this receipt before approval.
Credentials are mounted separately for authentication; they are not packet data
and must never be printed or copied into proposals or reports.

The packet includes `INSTRUCTIONS.md`, `EVALUATION.json`, the tool/skill catalogue,
public installed source access and explicitly selected reconstruction inputs.
Later packets contain the author's own preceding proposal, verdict, bounded
measured traces, and any retained nominal candidate. They exclude private manual
parameters, reference scans/poses and post-inference reference scores.

## Matched evaluation

Optional `evaluation.diagnostics` enables the same post-rollout audit for both
`continuous` and comparison-only `frozen_reference` backends. The latter supplies
`diagnostics_runtime` pointing to the corresponding continuous observation
configuration. No physics is stepped during this audit. It reconstructs kinematics
and recomputes contact forces from saved qpos/qvel/commands with zero warmstart.
These forces are estimates, not sensor measurements or original feedback logs.

The current backend evaluates a lifted, inverted placement and same-home return.
Its common audit adds explicit lift and airborne-inversion checks to the final
placement gate. Retention is measured on an independently defined airborne
interval, anchored at its first sample; rack-supported release is excluded.
Missing carrying evidence yields missing metrics, which fail admission limits.
Original controller-specific diagnostics remain unchanged and separately labeled.

Every rollout saves states, common diagnostic observations, object poses and a
compact verdict. Reference scorers consume these frozen outputs afterward,
outside the author and graph environments. A scorer should pin metric code,
reference inputs, body registration, symmetry and all time mappings. Report
retimed diagnostics separately from capture-time scores. Do not feed those
scores back into a campaign declared independent of evaluation reference data.

A numerical clock reset is a failed simulation, even if it also prevents the
posthoc audit from reading the trajectory. `record-clock-rejection --campaign
./campaign` recovers this specific reporting failure only when the trusted
simulator fault, graph failure and saved nonmonotonic timestamps agree. It archives
the original error and states, records task failure with unavailable audit metrics,
and allows the next proposal to receive that failure. It never reruns physics,
repairs a trajectory, changes limits or converts unrelated infrastructure errors.
Do not score or render the reset trajectory as a completed motion.

`variants.create_case` prepares bounded mass/inertia, sliding-friction and object
translation changes. Sliding friction is scaled on every compiled geom, including
unnamed ones, so the other surface cannot mask the change through pair mixing.
Torsional and rolling friction remain unchanged. Explicit contact-pair overrides
are rejected unless separately implemented. Validate each starting condition;
an initialization check is not a successful manipulation trial.

## Separate downstream gates

Declare independent numeric `limits` for `sft`, `rl` and `control`, whether each
requires all robustness cases, and named external `prerequisites`. Missing or
nonfinite metrics fail. Robustness requires exactly the declared case names and
successful, uninterrupted results. Infrastructure failures cannot become passes.

`simulation_criteria_passed` and `admitted` are separate: the latter also needs
explicit external evidence. SFT requires data/action-contract review; residual RL
requires the relevant adapter and zero-correction check; physical control needs
scene, sensing, mounting, hardware and independent-validation evidence appropriate
to the deployment. Policy edits cannot grant those prerequisites. No campaign
automatically launches training or physical control.

The search prioritizes nominal completion, then development-case robustness,
then its declared simulation target. Retention favors complete robustness,
number of successful declared cases, then passed simulation admission gates;
ties keep the earlier candidate. A later failed revision does not replace a
retained successful policy. Final independent cases should be evaluated after
selection, without becoming more author feedback. Finite deterministic case
counts are not population success-rate estimates.

For a narrow refinement that permits only constant command-vector changes,
`agentic_policy.literal_edits.bounded_keyword_vectors` checks existing named
keyword-list slots without executing the proposal. It enforces per-component
bounds, optional agreement between slots, and byte-identical source elsewhere,
including timing, guards and evaluation anchors. It does not replace geometry
validation, sandboxing or live collision/retention checks. Use this when a
campaign must preserve a retained clearance path while changing bounded pressure
targets; do not rely on author instructions alone to enforce that boundary.

## Human approval between broader revisions

A broader author may choose its controller edit strategy without changing the
trusted scene, command limits or evaluator. `author.human_review_between_proposals`
adds an explicit gate before every author call after the first evaluated proposal.
Campaign-wide data-transfer authorization is separate from outcome approval.
`authorization.review_requirement(state)` returns the next proposal ID, previous
proposal ID, previous result hash and campaign scope hash. Record these with
`approved: true` and the actual `user_decision_reference` in
`human-reviews/P<N>.json` only after the user approves that result. Missing or
stale receipts fail before an external author call; do not preapprove all rounds.

Keep nominal task completion, development robustness and SFT/RL/control admission
separate. Broader edits can replace policy-level diagnostic stops, but cannot
change trusted collision limits, evaluation anchors or the acceptance criteria.
The author receives its own measured feedback, including bounded
`feedback-environment.json` observations when available. Held-out evaluation and
private manual-controller parameters remain outside author inputs.

Verified review receipts are also copied into the next author workspace as
`HUMAN_REVIEW.json`, and the brief explicitly directs the author to that feedback.
This preserves the user's approved focus alongside numerical rollout feedback.
The note does not grant permission to alter the frozen scene, host limits or
assessment. Initial author calls have no preceding-outcome note.

## Match the author's static scene to the evaluated scene

Declare `author_scene` as a path inside the experiment bundle to require input
parity at campaign initialization. The check compares flattened MJCF structure,
all options/defaults (including collision flags), and referenced asset bytes;
asset relocation alone is allowed. Matching meshes is insufficient when solver
or collision options differ. A mismatch rejects preparation before an author
call. `scene_bundle.require_scene_input_parity` also supports explicit preflight
checks when using a pinned older campaign engine.

This certifies input parity only. Static checks must also use the matching host
implementation, initial state and runtime configuration. Record static admission
separately from live physics; a rejected approach can legitimately produce a
zero-duration trial, and its video must not imply executed motion. Repair stale
author inputs without modifying the frozen evaluated scene or relaxing limits.
