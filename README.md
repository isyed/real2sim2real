# real2sim2real

Composable tools for reconstructing human–object motion from RGB video,
repairing that reconstruction, retargeting it to a robot, and training a
correction controller in simulation. This repository is private for now.

| Module | Input | Output |
| --- | --- | --- |
| [reconstruction](modules/reconstruction/README.md) | RGB video, segmentation prompts, local pretrained models | Human motion, object mesh and poses, shared metric depth/camera, diagnostics |
| [reconstruction-repair](modules/reconstruction-repair/README.md) | Reconstruction and explicit image/depth evidence | Accepted repaired reference, retained parent, per-stage gate results |
| [retargeting](modules/retargeting/README.md) | Human/object reference, robot definition, body-model assets | Robot joint/root targets, object motion and geometric contact schedule |
| [controller-actor-training](modules/controller-actor-training/README.md) | Robot reference, robot assets and frozen controller weights | FlashSAC correction actor, training state and configuration |

The current implementations use SAM2, MoGe2, SAM 3D Body, depth-conditioned
SAM 3D Objects, FoundationPose, SOMA/Pink IK, and FlashCHORD/FlashSAC. The
initial robot recipe is G1 with Dex3 hands and SONIC. Alternative robots need
their own retargeting and controller/embodiment configuration.

## Choose the work you need

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/r2s2r modules
.venv/bin/r2s2r plan --config examples/full-pipeline.json
```

Planning loads no GPU models and starts no work. After configuring local
assets and installing the selected modules, run a chain:

```bash
.venv/bin/r2s2r run --config my-pipeline.json
# Or run just reconstruction from the same video configuration:
.venv/bin/r2s2r run --config my-pipeline.json --stages reconstruction
# Resume only completed, unchanged stages:
.venv/bin/r2s2r run --config my-pipeline.json --resume
```

To start from an existing artifact, set `input` to its directory and
`input_kind` to its kind. Use a new output directory when changing the selected
stages or parameters. Repair can be omitted; training can run independently
from a saved `robot-reference`. Examples are templates, not bundled datasets.

Each module is independently installable and may run in a separate Python
environment. That keeps PyTorch reconstruction dependencies separate from
JAX/CUDA training dependencies. Model weights, robot meshes, videos and replay
buffers are external assets. The pipeline has no model-download stage and
requests offline backend operation; provision all model caches before running.
Backend packages and model licenses must be installed/accepted separately.

## What is implemented and verified

The numerical repair engine, retargeter and FlashCHORD/FlashSAC source are
included, rather than wrappers around neighboring repositories. The new
orchestrator adds typed artifacts, content hashes, stage selection, deadlines,
optional GPU thermal supervision and resume checks.

CPU regression and packaging checks cover the extracted repair code and the
new module handoffs. **The new combined RGB-to-trained-actor entry point has
not yet completed a fresh GPU acceptance run.** Model/container installation
and physical grasp quality are separate checks; passing a repair gate does
not certify a grasp or a policy. See [validation](docs/VALIDATION.md).

- [Installation and local assets](docs/INSTALL.md)
- [Data contracts and coordinate systems](docs/DATA_CONTRACTS.md)
- [Architecture and future modules](docs/ARCHITECTURE.md)
- [Reproducibility and migration](docs/REPRODUCIBILITY.md)
- [Partial repair and locked-anchor diagnostics](docs/REPAIR_WINDOW_DIAGNOSTICS.md)
- [Upstream provenance and licenses](docs/THIRD_PARTY.md)

The repository contains research simulation tools. It does not yet contain
a hardware deployment module or a robot-specific real-world safety layer.

Reusable findings and their validation status are tracked in
[the module learning register](docs/LEARNINGS.md).
