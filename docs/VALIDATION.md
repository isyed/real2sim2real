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

## Isolated reconstruction workers

The subsequent dispatcher update passes 26 core/adapter CPU tests, with the native
Torch-dependent interop test skipped in the thin environment. Eight new tests
cover argument-array container prefixes, malformed commands, changed input
rejection, stopping on a failed phase, refusal to overwrite partial work, and
importing the dispatcher without Torch/NumPy/PyAV/trimesh. CPU-only offline imports
also pass in existing separate segmentation, mesh-generation and tracking
containers. Video decoding was exercised separately. These checks do not yet
establish full inference in the new phase layout.


The more complete worker check found a missing SAM3D Body source package named
`data` (transforms and batch preparation, not dataset assets). Those six files
are restored with original headers and hashes. All six full model CLI imports
now pass in offline containers with GPU access disabled. A rebuilt reconstruction
wheel was checked for the worker and restored transform/batch modules. The repair
suite currently passes 126 tests; the adapter suite passes 26 with one optional
Torch interop skip. Full reconstruction acceptance is in progress, not yet passed.

The first full execution completed RGB segmentation, MoGe2 depth, direct human
motion and human-aligned depth. It stopped when mesh generation selected the wrong
Torch model cache. The cache-routing fix passes its CPU regression check; both
required DINO weights load offline on CPU from the corrected cache. Mesh/tracking
and complete-artifact validation after that fix remain pending. The original run
and completed intermediate predictions are retained by the downstream caller.
