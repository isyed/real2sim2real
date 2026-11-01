# Configurable assistance and bounded progress

`controller_actor_training.residual.configuration` accepts optional
`assistance_thresholds`, one strictly increasing positive transition threshold
per existing curriculum stage. Defaults remain unchanged. For example:

```python
cfg = configuration(policy, output, environment_steps=30_000_000,
    assistance_thresholds=[4_000_000, 7_000_000, 10_000_000, 30_000_000])
```

The existing four stages use assistance scales 1.0, 0.5, 0.1 and 0.0. Under the
fixed curriculum's threshold semantics, the fourth threshold does not introduce
another stage. Each boundary rounds down to a complete parallel-world
collection, matching checkpoint budgets. With 256 worlds the nominal 10M
boundary is 9,999,872 transitions. Subsequent training is unassisted; evaluation
assistance must still be disabled explicitly. `residual.validate` reports both
thresholds and scales. Runtime and training curriculum copies must agree.

Changing a curriculum is a new experiment. Retain old checkpoints and document
whether a learner/replay is resumed or initialized fresh. An assistance-free
evaluation is distinct from assistance-free training.

`real2sim2real.progress_watchdog.run` bounds a process by both total wall time
and a caller-supplied progress token. Use completed transitions/frames rather
than arbitrary log growth when repeated warnings could hide a stall. On timeout
or interruption it reaps the child's process group. Admission, episode order,
spending limits and decisions to skip failures belong to the downstream runner.

Checkpoint relocation still rejects deployment copies by default. The explicit
`allow_deployment_chain=True` option permits a further path-only relocation after
validating the previous tensor-payload hash and original metadata. It preserves
the first training configuration and links the preceding deployment receipt.
This creates an inference copy; it does not make an actor-only file sufficient
for exact learner/replay continuation. Callers must independently verify asset
and reference identity before using a relocated actor.
