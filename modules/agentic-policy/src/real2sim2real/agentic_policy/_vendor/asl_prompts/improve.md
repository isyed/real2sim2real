# Improve the robot policy

The task:

> {{instruction}}

The policy in `workflow/` just ran on the training episodes. `FEEDBACK.md` says
how each episode ended and what happened in it; its first lines say what it
contains. Change the policy so more episodes succeed.

You may edit `workflow.json` and any script, add scripts, or restructure the
graph. Every tool must be in `TOOLS.md`, called by a literal name.

Find where each failed episode went wrong before you edit. The held-out
episodes place things differently again, so a fix that only fits the training
coordinates will not hold.
