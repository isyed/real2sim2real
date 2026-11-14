# Measured history, duplicate edits and stopping

New campaigns can opt into the repaired author loop in `experiment.json`:

```json
"author_loop": {
  "history": true,
  "stop_on_duplicate_edits": true,
  "baseline_edits": {}
}
```

`history` exports every archived proposal and its measured receipt into
`campaign-history/`, with an index and bounded contact-transition trace windows.
It does not search adjacent repositories or import held-out evaluation. Existing
explicitly supplied historical inputs remain separate. Changing the author-visible
scope requires a newly frozen configuration and applicable transfer authorization.

`stop_on_duplicate_edits` applies to the `reference_refinement` and
`contact_reference_refinement` backends. At the actual simulator RPC, the host
validates the edit and hashes its effective corrections on the frozen feedback
clock. Empty profiles and redundant constant-profile knots compare equally;
README or script changes cannot disguise the same effective edit. The host
compares against earlier evaluated nominal profiles in the same campaign. This
is scoped to identical frozen inputs, not a general equivalence checker for
arbitrary Python programs or a cross-scene result cache.

An optional `baseline_edits` supplies the exact previously verified baseline
profile to compare against, so its first repeated submission also stops before
physics. Declare it only after the baseline is verified. It is not an evaluation
result and does not grant acceptance. For the contact backend this must use the
structured contact/profile schema, not `{}`. Omitting it permits an initial
baseline verification trial.

A continuation may also supply `prior_edits`, a mapping from historical proposal
labels to validated edit objects. These are re-fingerprinted against the current
frozen inputs and checked at the same RPC boundary, with matches reported as
`prior:<label>`. This prevents an explicitly imported earlier experiment from
being repeated merely because the new campaign has no rounds yet. Include the
associated measured history separately; these declarations are not results and
must not be used to assert cross-scene outcome equivalence. Historical files are
never discovered or loaded implicitly. Duplicate detection must be enabled.

A duplicate stops the campaign with `duplicate_edit`, the matching proposal ID
and no physics execution. It does not rerun development cases, reuse a previous
success as a new success, replace the retained controller or silently consume
the remaining author budget. Invalid edits are rejected before fingerprinting.

An author can explicitly stop by removing `workflow.json` and submitting only:

```json
{"action": "retain_baseline", "reason": "No supported new hypothesis; inspect contact evidence."}
```

Save this as `candidate/decision.json`. A declared seed or retained successful
proposal must exist. Supplying both decision and workflow is rejected. This
outcome consumes the author submission, archives its reason, and stops without
physics or another author call. It does not certify the seed as successful.
Resume through a separately reviewed continuation, not by editing a frozen run.

Bounded candidate simulations may fail. The retained baseline stays immutable;
preserving nominal success and improving declared criteria are requirements for
promotion, not guarantees an untested proposal must make. Hard collision,
actuation, numerical and admission limits remain unchanged. Report distinct
effective controllers separately from author-call counts.

Tests: `modules/agentic-policy/tests/test_author_loop.py` covers history retention,
excluded unrelated files, transient contact evidence, explicit stops, failed
candidate retention, equivalent profiles and actual pre-step RPC rejection.
