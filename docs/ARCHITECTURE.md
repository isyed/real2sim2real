# Architecture and extension

```mermaid
flowchart LR
  V[RGB video and prompts] --> R[reconstruction]
  R --> P[reconstruction-repair]
  R --> T[retargeting]
  P --> T
  T --> A[controller-actor-training]
  A --> E[Independent unassisted evaluation]
```

The core package has no inference-library dependency. `plan` checks artifact
kinds and stage order. `run` writes a request and launches each stage in its
configured interpreter. Heavy libraries are imported in the stage process.
The supervisor controls the process group, including backend subprocesses.

Each stage directory contains its request, console log, artifact manifest and
successful execution receipt. Failed runs retain evidence and cannot silently
overwrite previous output. Resume checks the input hash, configuration,
artifact manifest hash and artifact contents.

The numerical engines live inside their owning modules. `r2s2r_engines` is
an internal namespace for reconstruction backends; applications should use
the four public stage adapters. FlashCHORD retains its upstream namespace to
keep its Hydra targets and checkpoint metadata consistent.

## Add a module

A new separately installed distribution can declare:

```toml
[project.entry-points."real2sim2real.stages"]
my-module = "my_module.registration:describe"
```

`describe()` returns `("my_module.stage", {"accepted-kind"}, "output-kind")`.
It must be lightweight and avoid GPU initialization. The executable module
accepts `--request PATH`; use `real2sim2real.stage.request_main` and
`real2sim2real.artifacts.publish`. Do not overwrite another module's name.
Install the descriptor in the orchestrator environment and the implementation
in the selected stage environment. Input/output contracts remain explicit.

Future camera-motion estimation, scene/support reconstruction, policy
evaluation and hardware deployment can be separate modules. An extension
point does not imply those capabilities are already implemented.
