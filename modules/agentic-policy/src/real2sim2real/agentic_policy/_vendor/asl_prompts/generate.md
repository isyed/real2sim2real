# Write a robot policy

Write a GaP workflow that does this task:

> {{instruction}}

`WORKFLOW_FORMAT.md` is the workflow format. `TOOLS.md` is the complete list of
tools the policy may call; nothing else exists when it runs.

Every episode places the objects differently and gives them a different mass
and friction. The policy runs on training episodes, whose results you will
read, and on held-out episodes you will never see. Read where things are with
the tools in each episode; do not write coordinates of one episode into the
policy.

Split the work into subgraphs along its natural stages, give every subgraph an
`on_error` exit, and map every exit at the top level.
