# Project boundaries

This project provides reusable, scenario-independent modules for video-to-robot
research. Keep dataset adapters, evaluation campaigns, cloud accounts and job
histories in downstream projects. Do not introduce runtime imports or symlinks
to neighboring repositories.

The modules are reconstruction, egocentric-reconstruction,
reconstruction-evaluation, reconstruction-repair, retargeting and
controller-actor-training. Optional heavy dependencies are loaded only by their
own stage. Preserve upstream copyright/license notices. Keep weights, robot
meshes, private videos and generated artifacts outside Git.

Use explicit units, coordinate transforms, timestamps and source hashes at
module boundaries. Rejected hypotheses must not silently replace their input.
GPU workloads require an explicit run request and the configured thermal and
time limits. CPU tests and configuration validation may run without GPU access.
Do not claim physical execution or complete model validation from CPU tests.

# Git workflow

Keep related iterations on one feature branch and reuse its existing pull
request. Create new pull requests only as part of a user-requested local/remote
sync. Local implementation and validation do not require opening another PR.

After the owner-approval policy is enabled, every main-branch merge requires
`@isyed`'s GitHub code-owner approval, including changes made by administrators.
Do not submit an approval on the owner's behalf or disable the protection to
complete a sync. GitHub disallows self-approval: use a separately authorized PR
author identity, or leave the PR pending and report that limitation. See
`CONTRIBUTING.md` for the review and branch-cleanup workflow.
