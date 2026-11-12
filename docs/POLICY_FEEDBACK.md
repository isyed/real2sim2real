# Measured feedback for a bounded policy revision

`policy_feedback` exports an explicit set of pinned development simulations for
an offline author. It does not run physics, update controller parameters, change
acceptance or expose every historical experiment. The source results, compiled
scenes, states and recursive evidence chain are verified before and after export.

The packet includes original verdicts, named correction parameters, first sampled
gate breaches, acquisition-anchored drift, pad forces versus pressure requests,
loaded/sliding sample fractions, friction ratios, wrist tracking and sampled
collision pairs. It reuses the existing contact/load-readiness diagnostics and
checks that reconstructed drift exactly agrees with the source's reported metric.

Retention uses the evaluator's cohort: acquisition completion through scheduled
release, with one fixed anchor. It never resets at the first hold sample or
includes intentional separation after release. Missing loaded-pad observations
remain missing sliding evidence, rather than becoming a false zero-slip result.

Contact forces and tangent velocities come from coherent saved-state solves.
Pressure targets and load estimates are preceding-step controller observations;
they are labelled separately. Planned pad gaps are not independent material slip
measurements. A friction ratio, load-readiness predicate or contact-centroid
movement is diagnostic evidence, not proof of a stable grasp. Negative signed
hand force during supported acquisition or placement is not by itself a failed
carrying verdict.

The optional MuJoCo read-only geometry replay identifies contact pairs at saved
qpos samples. It does not recover every unsaved physics substep or the precise
time of an original high-rate penetration peak. Original verdicts take precedence
when sparse contact distances do not show a recorded violation. No state in the
source run is changed. Arm margins and geometry-dependent limits come from the
pinned native audit/task, rather than a previous human-readable table.

```json
{
  "runs": [
    {"label": "frozen controller", "result": {"path": "/data/base/result.json", "sha256": "..."}},
    {"label": "previous proposal", "result": {"path": "/data/previous/result.json", "sha256": "..."}}
  ],
  "output": "/data/fresh-feedback"
}
```

```bash
python -m real2sim2real.controller_actor_training.policy_feedback \
  --config /data/feedback.json
```

`feedback.json` contains author-facing measurements without workspace paths;
`provenance.json` retains the input hashes and implementation hash for the campaign.
Run the exporter under the downstream campaign's finite CPU deadline. Inputs must
be complete continuous arm-replay results with existing contact instrumentation.
The current exporter reports the native replay's fixed 10 mm retention and
20 mm / 10 degree reference limits; it does not define a new task evaluator.

## Seed and underlying controller are different

A feedback-informed author may start from the previous candidate's graph. Its
parameters still refer to the original frozen controller. Replacing a previous
`+0.02 rad` offset with `+0.01 rad` means original plus `0.01`, never original plus
`0.03`. The validator's canonical zero graph remains unchanged. No correction
stacking or increased budget is implied by revision.

Stage a new packet with the declared feedback and previous graph; retain the
same writable `workflow/` boundary and existing checks. Freeze exactly one final
proposal before simulation. A failed admission or physical result must not cause
an undeclared repair round. Evaluate against both the previous proposal and the
original baseline, retaining all original criteria. Improved appearance or one
metric cannot promote a failed candidate.

This is offline feedback to the author. It is not an LLM in the 1 kHz controller
loop, a newly installed online feedback law, or residual RL training. The existing
servos remain responsible for all real-time simulation updates.
