"""Pinned-controller robustness cases; no controller selection or gate changes."""

import json
from pathlib import Path
import numpy as np
from real2sim2real.artifacts import digest


def validate_case(config):
    if "simulation_case" not in config:
        return None
    frozen = config["frozen_controller"]
    path = Path(frozen["result"])
    if digest(path) != frozen["sha256"]:
        raise ValueError("Frozen controller result changed")
    parent = json.loads(path.read_text())
    if parent.get("residual_adapter") is not None:
        raise ValueError("Residual policies need policy-aware robustness replay; do not silently omit the actor")
    if (
        not parent["success"]
        or digest(path.with_name("states.npz")) != parent["states_sha256"]
    ):
        raise ValueError("Robustness requires a pinned passing controller")
    allowed = {"output", "deadline_s", "simulation_case", "frozen_controller"}
    strip = lambda c: {k: v for k, v in c.items() if k not in allowed}
    if strip(config) != strip(parent["config"]):
        raise ValueError("Robustness may not change controller, path or timestep")
    case = config["simulation_case"]
    if set(case) != {"mass_scale", "sliding_friction", "initial_translation_m"}:
        raise ValueError("Explicit bounded robustness case required")
    shift = np.asarray(case["initial_translation_m"], float)
    if (
        shift.shape != (3,)
        or not np.isfinite(shift).all()
        or np.linalg.norm(shift) > 0.005
        or shift[2] != 0
        or not np.isfinite(case["mass_scale"])
        or not 0.5 <= case["mass_scale"] <= 2
        or (
            case["sliding_friction"] is not None
            and (
                not np.isfinite(case["sliding_friction"])
                or not 0.2 <= case["sliding_friction"] <= 1.2
            )
        )
    ):
        raise ValueError("Case exceeds mass/friction/initial horizontal pose bounds")
    return case


def apply_case(model, data, case):
    """Change only declared physical conditions before the first integration step."""
    if case is None:
        return
    import mujoco

    obj = model.body("task_object").id
    model.body_mass[obj] *= case["mass_scale"]
    model.body_inertia[obj] *= case["mass_scale"]
    if case["sliding_friction"] is not None:
        # Both contacting surfaces must change: MuJoCo combines their friction.
        model.geom_friction[:, 0] = case["sliding_friction"]
    q = int(model.joint("task_object_free").qposadr[0])
    data.qpos[q : q + 3] += case["initial_translation_m"]
    # mj_setConst uses qpos0 while rebuilding constants. Preserve the replay's
    # actual initialized grasp/arm state across this model update.
    qpos, qvel, ctrl, time = (
        data.qpos.copy(),
        data.qvel.copy(),
        data.ctrl.copy(),
        data.time,
    )
    mujoco.mj_setConst(model, data)
    data.qpos[:], data.qvel[:], data.ctrl[:], data.time = qpos, qvel, ctrl, time
    mujoco.mj_forward(model, data)
