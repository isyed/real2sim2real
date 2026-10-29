# Project boundaries

This project provides reusable, scenario-independent modules for video-to-robot
research. Keep dataset adapters, evaluation campaigns, cloud accounts and job
histories in downstream projects. Do not introduce runtime imports or symlinks
to neighboring repositories.

The four modules are reconstruction, reconstruction-repair, retargeting and
controller-actor-training. Optional heavy dependencies are loaded only by their
own stage. Preserve upstream copyright/license notices. Keep weights, robot
meshes, private videos and generated artifacts outside Git.

Use explicit units, coordinate transforms, timestamps and source hashes at
module boundaries. Rejected hypotheses must not silently replace their input.
GPU workloads require an explicit run request and the configured thermal and
time limits. CPU tests and configuration validation may run without GPU access.
Do not claim physical execution or complete model validation from CPU tests.
