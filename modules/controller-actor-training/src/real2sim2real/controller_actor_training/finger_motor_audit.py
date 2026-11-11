"""Read-only position-motor tracking/authority audit of saved robot states.

Reconstructs fixed-gain, affine-bias unit-gear joint motor effort algebraically.
This avoids a new contact solve or warmstart dependency. It is sampled actuator
effort, not contact force, pad pressure, or proof of a stable grasp.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from real2sim2real.artifacts import digest, write_json


def motor_state(command, position, velocity, gain, bias, force_range, control_range):
    values = [
        np.asarray(v, float)
        for v in (command, position, velocity, gain, bias, force_range, control_range)
    ]
    c, q, v, k, b, f, limits = values
    n = c.size
    if (
        not n
        or c.shape != (n,)
        or any(x.shape != (n,) for x in (q, v, k))
        or b.shape != (n, 3)
        or f.shape != (n, 2)
        or limits.shape != (n, 2)
        or any(not np.isfinite(x).all() for x in values)
        or (k <= 0).any()
        or (f[:, 0] >= 0).any()
        or (f[:, 1] <= 0).any()
        or (limits[:, 0] >= limits[:, 1]).any()
    ):
        raise ValueError(
            "Finite scalar position motor states and legal ranges required"
        )
    applied = np.clip(c, *limits.T)
    raw = k * applied + b[:, 0] + b[:, 1] * q + b[:, 2] * v
    force = np.clip(raw, *f.T)
    return dict(
        tracking_error_rad=applied - q,
        effort_nm=force,
        effort_limit_fraction=np.where(force >= 0, force / f[:, 1], force / f[:, 0]),
        effort_clipped=raw != force,
        command_clipped=c != applied,
    )


def run(source_path, output, *, deadline_s=60):
    import time
    import mujoco

    start = time.monotonic()
    implementation = Path(__file__).read_bytes()
    if not np.isfinite(deadline_s) or not 0 < deadline_s <= 120:
        raise ValueError("Motor audit requires a finite deadline up to 120 seconds")
    source_path = Path(source_path).resolve()
    source = json.loads(source_path.read_text())
    if source.get("schema") != "real2sim2real.arm-replay-pilot/1":
        raise ValueError("Saved arm physics required")
    for name, key in (("scene.xml", "scene_sha256"), ("states.npz", "states_sha256")):
        if digest(source_path.with_name(name)) != source[key]:
            raise ValueError("Motor audit artifacts changed")
    model = mujoco.MjModel.from_xml_path(str(source_path.with_name("scene.xml")))
    # Arm motors use a different controller and are explicitly excluded.
    aids = np.array(
        [
            a
            for a in range(model.nu)
            if not (model.actuator(a).name or "").startswith("pilot__")
        ],
        int,
    )
    joints = model.actuator_trnid[aids, 0]
    if not len(aids) or any(
        model.actuator_trntype[a] != mujoco.mjtTrn.mjTRN_JOINT
        or model.actuator_dyntype[a] != mujoco.mjtDyn.mjDYN_NONE
        or model.actuator_gaintype[a] != mujoco.mjtGain.mjGAIN_FIXED
        or model.actuator_biastype[a] != mujoco.mjtBias.mjBIAS_AFFINE
        or not np.array_equal(model.actuator_gear[a], [1, 0, 0, 0, 0, 0])
        or model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE
        or not model.actuator_forcelimited[a]
        or not model.actuator_ctrllimited[a]
        or model.actuator_biasprm[a, 1] != -model.actuator_gainprm[a, 0]
        for a, j in zip(aids, joints)
    ):
        raise ValueError(
            "Audit supports force/control-limited unit-gear hinge position motors only"
        )
    with np.load(source_path.with_name("states.npz"), allow_pickle=False) as archive:
        times, q, v, c = [archive[k] for k in ("timestamps_s", "qpos", "qvel", "ctrl")]
    records = source["records"]
    if (
        times.shape != (len(records),)
        or not np.isfinite(times).all()
        or np.any(np.diff(times) <= 0)
        or not np.array_equal(times, [r["time_s"] for r in records])
        or q.shape != (len(times), model.nq)
        or v.shape != (len(times), model.nv)
        or c.shape != (len(times), model.nu)
    ):
        raise ValueError(
            "Matching strictly increasing saved state/record clock required"
        )
    rows = []
    for i, t in enumerate(times):
        if time.monotonic() - start > deadline_s:
            raise TimeoutError("Saved motor audit deadline reached")
        values = motor_state(
            c[i, aids],
            q[i, model.jnt_qposadr[joints]],
            v[i, model.jnt_dofadr[joints]],
            model.actuator_gainprm[aids, 0],
            model.actuator_biasprm[aids, :3],
            model.actuator_forcerange[aids],
            model.actuator_ctrlrange[aids],
        )
        rows.append(
            dict(
                time_s=float(t),
                phase=records[i]["phase"],
                **{k: x.tolist() for k, x in values.items()},
            )
        )
    phases = {}
    for phase in dict.fromkeys(r["phase"] for r in rows):
        selected = [r for r in rows if r["phase"] == phase]
        phases[phase] = dict(
            samples=len(selected),
            first_s=selected[0]["time_s"],
            last_s=selected[-1]["time_s"],
            joints={
                model.joint(j).name: dict(
                    maximum_tracking_error_rad=max(
                        abs(r["tracking_error_rad"][k]) for r in selected
                    ),
                    maximum_effort_limit_fraction=max(
                        r["effort_limit_fraction"][k] for r in selected
                    ),
                    effort_clipped_fraction=float(
                        np.mean([r["effort_clipped"][k] for r in selected])
                    ),
                    command_clipped_fraction=float(
                        np.mean([r["command_clipped"][k] for r in selected])
                    ),
                )
                for k, j in enumerate(joints)
            },
        )
    report = dict(
        schema="real2sim2real.finger-motor-audit/1",
        source=str(source_path),
        source_sha256=digest(source_path),
        states_sha256=source["states_sha256"],
        scene_sha256=source["scene_sha256"],
        source_success=source["success"],
        source_rejected_for=source["rejected_for"],
        diagnostic_only=True,
        acceptance_changed=False,
        observation="Saved-state affine position-motor effort; no contact solve or physics step",
        joint_names=[model.joint(j).name for j in joints],
        phases=phases,
        records=rows,
    )
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    (out / "finger_motor_audit.py").write_bytes(implementation)
    report["implementation_sha256"] = digest(out / "finger_motor_audit.py")
    write_json(out / "audit.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deadline-s", type=float, default=60)
    args = parser.parse_args()
    run(args.result, args.output, deadline_s=args.deadline_s)


if __name__ == "__main__":
    main()
