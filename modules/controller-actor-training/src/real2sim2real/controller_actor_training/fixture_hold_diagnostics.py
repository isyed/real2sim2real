"""Separate saved fixture tracking error from acquisition-anchored grasp drift."""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from real2sim2real.artifacts import digest, write_json


def audit(directory):
    import mujoco
    from .fixture_hold import Commands
    from .paired_acquisition_audit import pinned_json
    from .object_grasp_pilot import (
        load_scene,
        load_plan,
        set_root_frame,
        object_transform,
    )

    directory = Path(directory)
    result = json.loads((directory / "result.json").read_text())
    if (
        result["schema"] != "fixture-hold/1"
        or not result["physics_run"]
        or digest(directory / "states.npz") != result["states_sha256"]
    ):
        raise ValueError("Unchanged saved fixture physics required")
    config = result["config"]
    paired = pinned_json(config["acquisition_audit"])
    plan = pinned_json(paired["config"]["plan"])
    source = pinned_json(plan["config"]["source_trial"])
    model, data, _ = load_scene(result["configuration"])
    settled = load_plan(result["configuration"])["settling"]
    hands = [result["candidate"], result["candidate"]["support"]]
    for h in hands:
        set_root_frame(
            model,
            data,
            h["root_body"],
            h["fixture_prefix"],
            h["base_position_m"],
            h["base_rotation"],
            preserve_state=True,
        )
    command = Commands(
        model,
        source,
        hands,
        settled,
        config["condition"],
        config.get("acquisition_blend_s", 0),
    )
    with np.load(directory / "states.npz") as f:
        q, times = f["qpos"].copy(), f["timestamps_s"].copy()
    if len(q) != len(result["records"]):
        raise ValueError("Saved sample count changed")
    errors = {}
    for row, state, t in zip(result["records"], q, times):
        if abs(row["time_s"] - t) > 1e-9:
            raise ValueError("Saved clocks differ")
        data.qpos[:] = state
        mujoco.mj_kinematics(model, data)
        command.at(t)
        mujoco.mj_kinematics(model, command.scratch)
        values = []
        for h in hands:
            root = model.body(h["root_body"]).id
            values.append(
                float(np.linalg.norm(data.xpos[root] - command.scratch.xpos[root]))
            )
        actual = object_transform(model, data)
        goal = command.goal(t)
        values += [
            float(np.linalg.norm(actual[:3, 3] - goal[:3, 3])),
            float(
                np.rad2deg(
                    Rotation.from_matrix(actual[:3, :3] @ goal[:3, :3].T).magnitude()
                )
            ),
        ]
        errors.setdefault(row["phase"], []).append(values)
    names = [
        "primary_wrist_position_error_m",
        "support_wrist_position_error_m",
        "object_position_error_m",
        "object_rotation_error_deg",
    ]
    phases = {
        p: {
            name: dict(maximum=float(np.asarray(v)[:, i].max()), final=float(v[-1][i]))
            for i, name in enumerate(names)
        }
        for p, v in errors.items()
    }
    output = dict(
        schema="fixture-hold-diagnostics/1",
        result_sha256=digest(directory / "result.json"),
        states_sha256=result["states_sha256"],
        phase_errors=phases,
        interpretation="Wrist errors compare actual fixture FK with nominal commanded FK. Object errors use this diagnostic goal, not full demonstrated-motion scores. No simulation rerun or evidence promotion.",
    )
    write_json(directory / "tracking-diagnostics.json", output)
    return output


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory", type=Path, required=True)
    p.add_argument("--worker", action="store_true")
    a = p.parse_args()
    if a.worker:
        audit(a.directory)
    else:
        from real2sim2real.bounded_process import run

        run(
            [
                sys.executable,
                "-m",
                __spec__.name,
                "--directory",
                str(a.directory.resolve()),
                "--worker",
            ],
            timeout=120,
            env={
                **os.environ,
                "CUDA_VISIBLE_DEVICES": "",
                "OPENBLAS_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
            },
        )
