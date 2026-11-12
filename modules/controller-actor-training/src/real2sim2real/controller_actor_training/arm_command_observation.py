"""Record exact kinematic arm targets separately from servo feedforward terms."""

import numpy as np


def capture(model, actual, scratch, roots, qids, command):
    import mujoco

    result = {
        k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in command.items()
    }
    result["sample_time_s"] = float(actual.time)
    result["wrists"] = {role: {} for role in roots}
    for label, q in [
        ("planned", command["planned_q_rad"]),
        ("commanded", command["kinematic_q_rad"]),
    ]:
        scratch.qpos[:] = actual.qpos
        scratch.qpos[qids] = q
        mujoco.mj_kinematics(model, scratch)
        for role, body in roots.items():
            for name, data in [(label, scratch), ("actual", actual)]:
                pose = np.eye(4)
                pose[:3, :3] = data.xmat[body].reshape(3, 3)
                pose[:3, 3] = data.xpos[body]
                result["wrists"][role][name] = pose.tolist()
    return result
