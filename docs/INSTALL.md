# Installation

Use Linux and Python 3.11 for the existing heavy backends. The orchestrator
also supports Python 3.12. Keep independent environments for reconstruction,
CPU repair, retargeting, and controller training.

From this repository root:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -e .

python3.11 -m venv .venv-repair
.venv-repair/bin/python -m pip install -e . -e modules/reconstruction-repair

python3.11 -m venv .venv-retarget
.venv-retarget/bin/python -m pip install -e . -e modules/retargeting \
  -r modules/retargeting/requirements-runtime.txt

python3.11 -m venv .venv-training
.venv-training/bin/python -m pip install -e . -e modules/controller-actor-training \
  -r modules/controller-actor-training/requirements-gpu.txt
```

The training lock uses CUDA 13 JAX wheels, Warp/Newton, and ONNX Runtime GPU.
Check your driver before installing that environment. Do not mix these pins
with the reconstruction environment's PyTorch 2.8/CUDA 12.8 stack.

Reconstruction needs native rendering extensions and separately licensed
models. See [its module installation](../modules/reconstruction/README.md)
and build recipe. Installing the small Python distribution alone does not
install all GPU backends. Build images and download models explicitly; no
build or model download runs when inspecting or planning a pipeline.

## Asset locations

Configure paths in your private pipeline JSON; paths are relative to that
JSON's directory unless absolute. No sibling checkout is required.

```
model-store/
  sam2/sam2.1_hiera_large.pt
  sam3d/checkpoints/pipeline.yaml       # plus all referenced weights
  reconstruction/
    sam3d_body/checkpoints/sam-3d-body-dinov3/model.ckpt
    sam3d_body/checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt
    sam3d_body/torch_home/             # locally cached image encoder
    foundationpose/nvlabs_pytorch/     # score/refine models and configs
    hf_home/                          # pinned MoGe caches
robot-assets/
  body_models/soma/                    # SOMA and MHR inversion assets
  urdfs/g1/main_with_hand.urdf          # complete mesh/package closure
  policies/sonic/                      # frozen encoder/decoder policies
```

The SONIC adapter lists required filenames in
`flash_chord/runtime/actions/sonic.py`. SOMA conversion checks required assets
and rejects unresolved Git LFS pointers. Set `R2S2R_ASSETS` when invoking
retargeting or training directly; the stage adapters set it from `assets`.
Custom retargeting configurations use `asset://...` paths under this root,
or paths relative to their own configuration directory.

## Run controls

Set `execution.python`, `timeout_seconds`, and, for a GPU stage,
`thermal: {"pause_c":82,"resume_c":74,"stop_c":83}`. GPU supervision uses
`nvidia-smi`; telemetry failures stop the process group. It does not change
power or clock limits. Set hardware limits through your administrator before
launching a GPU workload. CPU repair should use `execution.cpu_only: true`.

The adapter never launches cloud resources. Training is an explicit `run`
action. A step budget is rounded **down** to a whole parallel-world collection;
the learner also requires enough steps to fill its configured replay warmup.

For development, install `.[test]` and the module dependencies, then run
`python -m pytest`. The repository excludes environments, weights, videos,
artifacts and training replay from Git.
