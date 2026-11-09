# Testing a retained actor after changing object inertials

Better isolated settling does not establish better robot control. Evaluate
these claims separately: object-only stability, retained-actor inference, and
learning under the changed dynamics.

`real2sim2real.retargeting.object_inertia.audit_inertial_pair` verifies that two
single-link rigid-object URDFs differ only in inertial properties. It requires
fixed positive mass, finite centre of mass, positive physically valid inertia,
identical non-inertial XML and identical referenced mesh bytes. It does not
establish true material distribution or physical stability.

Prepare two self-contained policy bundles with identical controller settings,
motion, support and robot assets. Use explicit deployment copies of the same
actor, retaining matching tensor-payload hashes. `relocate_checkpoint` permits
declared path changes and requires caller evidence: record the deliberate
inertial-model change there rather than describing it as an identical scene.
`evaluate_actor` records the checkpoint hash and deployment receipt in its audit.

Evaluate both models from the same initial state without assistance, using one
producer build. Record a primary comparison in advance and repeat promising
results. Check task scores, balance, palm tracking and sustained physical contact;
do not promote a candidate solely from an isolated stability check.

Inference transfer does not validate resuming old optimizer/replay state. A
candidate that settles better but degrades the retained actor may still warrant
a separately authorized fresh learner. Preserve the original pairing and all
failed trials while testing that hypothesis.

`controller_actor_training.inertial_ablation.package` creates a separate portable
variant from an existing policy and a proposed object URDF. It runs the strict
inertial audit, preserves controller configuration and all other assets, updates
nested reference/policy hashes and records the modelling evidence. With hard links,
changed files and receipts are replaced independently so the source stays intact.
The output remains physically unvalidated until the separate GPU tests complete.

Contact retention checks must declare which hands the task uses. Do not inherit
a right-hand-only rule from a single-hand experiment into a two-hand task. Report
each hand separately as well as the declared aggregate; summed contact counts
are hand-frame counts and can count a simultaneous frame twice. Keep losses on
one hand explicit even when aggregate contact increases.

Separate a score-improving partial candidate from a candidate that passes every
balance/contact gate. Preserve all repeat results and show failed executions.
An explicit partial nomination may be useful for further experiments, but it
does not establish robust manipulation or silently override a failed gate.
