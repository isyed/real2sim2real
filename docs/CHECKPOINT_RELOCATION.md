# Inference checkpoint path relocation

`controller_actor_training.relocate_checkpoint.relocate` writes a fresh deployment
copy of a safetensors checkpoint while retaining its tensor descriptors and raw
tensor payload byte for byte. It permits only explicit absolute paths for the
reference parquet and controller policy directory. Each replacement must match
the expected old value. The original resolved training configuration remains in
metadata alongside a deployment receipt and source/payload hashes.

This is not retraining or evidence that an actor transfers successfully. Before
relocation, the caller must verify reference motion, timing, action semantics,
robot/controller asset hashes and any deliberately changed scene dynamics. Keep
the original checkpoint immutable. Run an actual finite actor forward pass on CPU
before requesting a physics rollout. Then evaluate with assistance disabled,
frame-zero initialization, no automatic reset and explicit balance/contact gates.
Changing an object's physical model must be disclosed even if its reference
trajectory and controller weights are unchanged.
