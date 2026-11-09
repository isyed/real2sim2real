# Matched frozen and learned evaluation cohorts

`controller_actor_training.evaluation_build` identifies the complete evaluator,
including both frozen and actor branches, the runtime and source-clock export.
Both branches use the same build identity on newly generated trajectories.
Existing trajectories must never have their provenance rewritten to match it.

`candidate_selection.choose` compares finite repeated cohorts with the same
reference/model/build identity. AUC is higher-is-better; position error is
lower-is-better. Its configurable selection policy can admit a success gain
with a bounded AUC loss. The caller supplies independently scored metrics after
rollouts; neither clean targets nor these scores become controller inputs.
Three trials are required, and trial one remains the primary output. This is a
selection rule, not a statistical significance test.

`retargeting.static_support` converts explicitly selected predicted depth into
static convex collision surfaces. It excludes foreground masks, rejects
insufficient or degenerate evidence, and leaves coordinate transforms and
ROI selection to the caller. It does not invent thickness or a clean scene.

`bounded_process.run` gives a pipeline stage a wall-clock limit and terminates
its process group on timeout/interruption. GPU thermal policies and cloud
spending limits belong to the enclosing application.
