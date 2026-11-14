# Testing the reusable modules

Run from the repository root with Python 3.11 or 3.12:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -e '.[test]' ruff
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m pytest -q
.venv/bin/ruff check --select F --extend-exclude '**/_vendor/**' \
  src/real2sim2real modules/*/src/real2sim2real tests \
  modules/reconstruction-repair/tests modules/agentic-policy/tests
.venv/bin/python scripts/check_source.py
git diff --check
```

GitHub Actions runs the CPU suite and first-party undefined-name/import checks
on Python 3.11 and 3.12. The commands above also check test-source lint and private
workspace coupling locally. Pytest adds the module source directories
to its import path. Synthetic fixtures require no private repositories, robot
assets, model weights or GPUs. Optional libraries report explicit skips; a skip
is not simulation or model validation. Keep test module names unique across the
three test directories. See the [coverage catalogue](TEST_COVERAGE.md) for the
behavior checked by each suite.

Pinned upstream sources are excluded from first-party lint. Bundled hash tests
check their integrity; documented integration patches update the provenance
manifest and refresh script together. Verbatim skill reference scripts remain
unchanged and are not automatically executable robot adapters.

## Optional physics checks

Install compatible MuJoCo and numerical dependencies in a separate runtime,
then run the relevant tests against the same repository sources, for example:

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MUJOCO_GL=disable \
  /path/to/runtime/bin/python -m pytest -q \
  tests/test_loaded_posture.py tests/test_loaded_contact_repair.py \
  tests/test_contact_conditioned_posture.py tests/test_continuous_policy.py \
  modules/agentic-policy/tests
```

FlashCHORD/actor checks instead need their compatible Newton, Warp,
JAX/Flax/Optax and checkpoint libraries. Torch hand/body interoperability needs
its own retargeting environment. Optional runtime results must identify skipped
tests and dependencies. GPU inference and training require separately authorized,
bounded runs.

## Standalone agentic-policy acceptance

Build and install the root and selected module wheels as described in
[agentic-policy](../modules/agentic-policy/README.md). Run its
`tools/clean_acceptance.py` inside an outer sandbox with no workspace mounts.
It checks installed code/skill access, campaign history, manual/review modes and
isolated graph execution using a synthetic author; it does not call an external
model or demonstrate a physical task.

The independent `tools/manual_replay_acceptance.py` compares a retained successful
controller against complete archived simulation states. A meaningful comparison
must preserve physics, timing, acquisition, transport, placement and release.
See [frozen reference replay](FROZEN_REFERENCE_REPLAY.md). Complete zero-correction
reproduction is required before evaluating a new adaptive controller interface.

Passing static fitting or CPU tests does not admit a grasp, training or physical
control. Keep nominal completion, robustness and SFT/RL/control criteria separate;
see [control readiness](CONTROL_READINESS.md).
