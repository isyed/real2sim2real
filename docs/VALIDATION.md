# Extraction validation

Validation date: 2026-10-29. No GPU workload, model download or cloud instance
was started for these checks.

| Check | Result |
| --- | --- |
| Portable repair regression suite | 125 passed |
| Core/stage adapter suite, including actual numeric repair and subprocess resume | 19 additional tests passed |
| Native retargeting-to-FlashCHORD Parquet relocation | Passed separately in the Torch retargeting environment; skipped in the thin CPU environment |
| Source syntax and sibling-workspace import audit | Passed |
| Core plus four module wheels | All five built; source/config/assets-code payloads checked |
| Wheel imports from a neutral directory | Checked without a source checkout on the import path |

The checks include invalid stage chains, overwritten outputs, artifact hash
changes, invalid deadlines, shared depth/camera reprojection, training-budget
rounding, protected Hydra overrides, early assistance removal, native human
payload preservation and object paths after relocating a robot reference.
They exposed and fixed partition metadata loss when flattening a Parquet
artifact.

Not yet established by this extraction:

- A fresh build of the combined reconstruction image with selected upstream
  GPU backend commits and all licensed model assets.
- A full new RGB clip through the combined reconstruction/repair/retargeting
  adapter and subsequent physical simulation.
- Performance, memory use, grasp quality or training throughput of the new
  package layout on a particular GPU.

The code is a modular extraction with CPU-verified handoffs. Its original
backend implementations have been used in prior research workflows, but that
history is not a substitute for validating this new combined entry point.
