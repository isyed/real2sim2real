# Rewards for interactions that do not require a grasp

A pushing task can use a foot, forearm or another robot surface. Requiring hand
contact to admit every task rejects valid interactions. Declare the expected
effector, bind source-geometry proximity evidence to the exact input/reference,
and enable its robot/object collisions with `contact_admission.assess`.
Admission remains separate from object settling and unassisted execution.

Object position/orientation rewards score the result of a push regardless of the
effector. Whole-body and pelvis tracking provide motion/posture guidance. The
current ReconBody objective has no dedicated nonhand-contact reward. Its hand
force-closure bonus sees configured hand contacts only; it does not measure foot
contact. Palm tracking follows the recorded posture and is not itself a grasp
requirement. The progress term is episode-time fraction, not measured object progress.

Geometric hand proximity can occur incidentally even in a nonhand task. Audit
the actual controller-rate labels rather than assuming every hand phase is zero.
A zero force-closure term is not a negative grasp penalty. Existing hand-contact
counters must not be reported as all-body task-contact measurements.

```python
from real2sim2real.controller_actor_training import residual, reward_alignment

config = residual.configuration(policy, output, environment_steps=5_000_000)
audit = reward_alignment.audit(
    config, hand_contact_active,  # binary [hand, control_frame]
    expected_effectors=["left_foot"], control_fps=50.0,
)
# Optional, separately versioned experiment, before starting a fresh learner:
config = reward_alignment.without_hand_grasp_reward(
    config, expected_effectors=["left_foot"],
)
residual.validate(config)
```

The opt-in profile copies its input and sets hand force closure to zero in the
objective, environment objective and every curriculum stage. It retains body,
palm and object rewards and the geometric phase labels. Hand/gripper tasks,
missing nonhand collision scope, and checkpoint/replay resumes are rejected.
It does not add a foot reward or establish training/task success. Keep an
unchanged first pilot when possible; compare any task-specific shaping as a
new experiment rather than changing a running critic/replay history.
