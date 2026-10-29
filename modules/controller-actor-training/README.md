# Controller and actor training

This module contains FlashCHORD simulation and the actual FlashSAC learner,
replay buffer, checkpoints and unassisted evaluation code.

The G1 recipe combines three components:

- **Frozen SONIC:** maps the body reference and robot state to body joint
  targets. It is a pretrained motion controller, not a task-language planner.
- **Fixed hand closing:** independently opens/closes left and right fingers
  from reference contact phase and actual palm proximity to the current
  object. Targets are rate-limited and bounded by joint limits.
- **Learned residual actor:** FlashSAC learns body and finger corrections on
  top of those targets. Zero residuals do not mean zero finger targets when
  the fixed hand controller is enabled.

FlashCHORD supplies parallel physics, observations and reward terms.
FlashSAC collects transitions into a bounded ring replay buffer, samples
minibatches, fits critics and updates the actor. Critic and replay state are
used for training; actor inference does not require the critic or replay.

The default generic `g1_bounded_flash_sac` recipe enables fixed hand closing,
uses 256 worlds and a one-million-transition replay capacity. It reduces
object assistance at 2M/4M transitions and removes it at 6M, with frame-zero
starts introduced early. These are experiment defaults, not established
success guarantees. Default rewards track pelvis/body/palms/object position,
encourage force closure/progress and penalize termination and rough actions.
Object orientation is an available objective option, not a hidden critic
input or automatically enabled additional reward.

Set an explicit `environment_steps` budget. The adapter rounds it down to a
multiple of the world count. `checkpoint_every` controls collection intervals.
Reaching a checkpoint does not select it for deployment: evaluate without
object assistance and compare multiple independent rollouts.

`scene.support` is false in the generic bounded recipe. Enable it explicitly
only with a verified robot-aligned `support.usda`. Tabletop tasks usually need
such a surface; the reconstruction module does not currently create it.

## Independent use

Use `examples/training.json` with a saved `robot-reference` artifact. Native
Hydra entry points remain available:

```bash
export R2S2R_ASSETS=/absolute/path/to/robot-assets
python -m real2sim2real.controller_actor_training.train --cfg job \
  experiment=g1_bounded_flash_sac
python -m real2sim2real.controller_actor_training.evaluate \
  evaluation.checkpoint=/absolute/path/to/policy_9999872.safetensors \
  evaluation.reference_artifact=/absolute/path/to/robot-reference \
  evaluation.world_count=1 logging.mode=disabled
```

The evaluation command launches physics and should use the same GPU
temperature/time supervision as training. Calling `--cfg job` only prints
configuration. Logging uploads are disabled by default.

For exact continuation, retain learner state and optionally replay via the
native `training.resume`, `training.checkpoint_path` and
`training.checkpoint.load_replay_path` settings. Saving replay is off by
default. Rebuilding a fresh buffer is possible, but changes the experiment.

The GPU stack is pinned in `requirements-gpu.txt`. No pretrained weights,
episode-specific actors, datasets or cloud launchers are embedded here.
