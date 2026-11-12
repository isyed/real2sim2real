"""Read-only decomposition of planned motion, command changes and tracking error."""

import argparse
import html
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from real2sim2real.artifacts import digest, write_json

from .paired_acquisition_audit import pinned_json
from .sampled_motion_variation import position_variation


def decompose(planned, commanded, actual):
    p, c, a = [np.asarray(v, float) for v in (planned, commanded, actual)]
    if (
        p.ndim != 2
        or p.shape[1] != 3
        or c.shape != p.shape
        or a.shape != p.shape
        or not all(np.isfinite(v).all() for v in (p, c, a))
    ):
        raise ValueError("Aligned finite world positions required")
    return {
        "command_change_m": c - p,
        "tracking_error_m": a - c,
        "total_error_m": a - p,
    }


def motion_summary(records, window):
    rows = [
        r
        for r in records
        if window["start_s"] - 1e-8 <= r["time_s"] <= window["end_s"] + 1e-8
    ]
    times = [r["time_s"] for r in rows]
    result = {}
    for role in rows[0]["arm_command"]["wrists"]:
        poses = {
            key: np.array([r["arm_command"]["wrists"][role][key] for r in rows])
            for key in ("planned", "commanded", "actual")
        }
        delta = decompose(
            *(poses[key][:, :3, 3] for key in ("planned", "commanded", "actual"))
        )
        stats = {}
        for key, values in delta.items():
            norm = np.linalg.norm(values, axis=1) * 1000
            stats[key] = {
                "rms_mm": float(np.sqrt(np.mean(norm**2))),
                "peak_mm": float(norm.max()),
                "variation": position_variation(times, values),
            }
        stats["sampled_variation"] = {
            key: position_variation(times, pose[:, :3, 3])
            for key, pose in poses.items()
        }
        angle = np.rad2deg(
            Rotation.from_matrix(
                poses["actual"][:, :3, :3]
                @ poses["commanded"][:, :3, :3].transpose(0, 2, 1)
            ).magnitude()
        )
        stats["actual_command_rotation_error_deg"] = {
            "rms": float(np.sqrt(np.mean(angle**2))),
            "peak": float(angle.max()),
        }
        result[role] = stats
    return result


def force_samples(result, path, start_s, end_s):
    import mujoco

    from .arm_replay_pilot import subtree_geoms
    from .support_control import contact_wrench_on_body
    from .table_load_transfer import observe_table_force

    m = mujoco.MjModel.from_xml_path(str(path.with_name("scene.xml")))
    d = mujoco.MjData(m)
    obj = m.body("task_object").id
    objects = set(np.flatnonzero(m.geom_bodyid == obj).tolist())
    table = m.geom("table").id
    audit = json.loads(Path(result["config"]["audit"]).read_text())
    if digest(Path(result["config"]["audit"])) != result["config"]["audit_sha256"]:
        raise ValueError("Pinned arm audit changed")
    hands = set()
    for arm in audit["config"]["arms"]:
        hands.update(subtree_geoms(m, m.body(arm["root"]).id))
    with np.load(path.with_name("states.npz"), allow_pickle=False) as states:
        rows = []
        for i, t in enumerate(states["timestamps_s"]):
            if not start_s - 1e-8 <= t <= end_s + 1e-8:
                continue
            d.qpos[:], d.qvel[:], d.ctrl[:] = (
                states["qpos"][i],
                states["qvel"][i],
                states["ctrl"][i],
            )
            d.qacc_warmstart[:] = 0
            d.time = float(t)
            mujoco.mj_forward(m, d)
            f, count = observe_table_force(m, d, objects, table, obj)
            hand = np.zeros(3)
            for cid, c in enumerate(d.contact):
                pair = {int(c.geom1), int(c.geom2)}
                if pair & objects and pair & hands:
                    hand += contact_wrench_on_body(m, d, cid, obj)[0]
            rows.append(
                {
                    "time_s": float(t),
                    "table_vertical_force_n": f,
                    "hand_vertical_force_n": float(hand[2]),
                    "table_contacts": count,
                }
            )
    supported = [
        r for r in rows if r["table_contacts"] and r["table_vertical_force_n"] > 0.1
    ]
    post_touch = [
        r for r in rows if supported and r["time_s"] >= supported[0]["time_s"]
    ]
    return {
        "samples": rows,
        "weight_n": float(m.body_mass[obj] * 9.81),
        "post_touch_window_start_s": supported[0]["time_s"] if supported else None,
        "post_touch_loaded_table_fraction": len(supported) / len(post_touch)
        if post_touch
        else None,
        "post_touch_loaded_sample_count": len(supported),
        "post_touch_sample_count": len(post_touch),
        "mean_table_force_n": float(
            np.mean([r["table_vertical_force_n"] for r in supported])
        )
        if supported
        else None,
        "peak_table_force_n": max(
            (r["table_vertical_force_n"] for r in supported), default=None
        ),
        "mean_downward_hand_force_n": float(
            np.mean([max(0, -r["hand_vertical_force_n"]) for r in supported])
        )
        if supported
        else None,
        "coherent_saved_state_solves": True,
        "physics_replayed": False,
    }


def run(config):
    out = Path(config["output"])
    out.mkdir(parents=True, exist_ok=False)
    cases = []
    for entry in config["cases"]:
        r = pinned_json(entry["result"])
        path = Path(entry["result"]["path"])
        for filename, key in [
            ("scene.xml", "scene_sha256"),
            ("states.npz", "states_sha256"),
        ]:
            if digest(path.with_name(filename)) != r[key]:
                raise ValueError("Physical command archive changed")
        if (
            not r.get("physics_run")
            or "incomplete_pilot" in r["rejected_for"]
            or not all("arm_command" in row for row in r["records"])
        ):
            raise ValueError(
                "Complete physical run with exact command telemetry required"
            )
        reconstruction = 0.0
        clock_age = []
        for row in r["records"]:
            c = row["arm_command"]
            kp = r["config"]["kp_nm_per_rad"]
            kd = r["config"]["kd_nm_s_per_rad"]
            servo = (
                np.array(c["kinematic_q_rad"])
                + (
                    np.array(c["model_bias_nm"])
                    + np.array(c["inertial_feedforward_nm"])
                    + kd * np.array(c["velocity_feedforward_rad_s"])
                )
                / kp
            )
            reconstruction = max(
                reconstruction, float(np.max(abs(servo - c["servo_ctrl_rad"])))
            )
            clock_age.append(row["time_s"] - c["command_time_s"])
        if (
            reconstruction > 1e-12
            or min(clock_age) < -1e-8
            or max(clock_age) > r["controller_period_s"] + 1e-7
        ):
            raise ValueError(
                "Command/feedforward reconstruction or clock alignment failed"
            )
        cases.append(
            {
                "label": entry["label"],
                "result": entry["result"],
                "windows": {
                    w["label"]: motion_summary(r["records"], w)
                    for w in config["windows"]
                },
                "force": force_samples(
                    r, path, config["force_start_s"], config["force_end_s"]
                ),
                "maximum_servo_reconstruction_error_rad": reconstruction,
                "maximum_command_age_s": max(clock_age),
                "success": r["success"],
                "rejected_for": r["rejected_for"],
                "primary_drift_mm": r["hold_slip_m"] * 1000,
                "support_drift_mm": r["support_wrist_drift_m"] * 1000,
                "minimum_clearance_mm": r["hold_minimum_clearance_m"] * 1000,
                "continuous_metrics": r["continuous_metrics"],
                "table_load_transfer": r.get("table_load_transfer"),
            }
        )
    summary = {
        "schema": "arm-command-audit/1",
        "config": config,
        "cases": cases,
        "physics_replayed": False,
        "controller_promoted": False,
        "implementation_sha256": digest(Path(__file__)),
    }
    write_json(out / "summary.json", summary)
    parts = [
        '<!doctype html><meta charset="utf-8"><title>Command tracking and table load transfer</title><style>body{font:16px system-ui;max-width:1150px;margin:30px auto;padding:15px}table{border-collapse:collapse}td,th{border:1px solid #bbb;padding:8px;text-align:left}</style><h1>Command tracking and table load transfer</h1><p>Exact planned, corrected kinematic and actual wrist poses are separated. Servo targets include model bias and velocity/inertial feedforward and are not interpreted as geometric wrist targets. Forces below are coherent read-only solves at saved states. No acceptance criteria changed; no candidate is promoted by this report.</p><table><tr><th>Case</th><th>Mean / peak table force, N</th><th>Mean downward hand force, N</th><th>Loaded table fraction after touchdown</th><th>Primary / support drift, mm</th><th>Clearance, mm</th><th>Release</th></tr>'
    ]
    for c in cases:
        f = c["force"]
        fmt = lambda v: "unavailable" if v is None else f"{v:.2f}"
        fraction = f["post_touch_loaded_table_fraction"]
        coverage = "unavailable" if fraction is None else f"{100 * fraction:.1f}%"
        parts.append(
            f"<tr><td>{html.escape(c['label'])}</td><td>{fmt(f['mean_table_force_n'])} / {fmt(f['peak_table_force_n'])}</td><td>{fmt(f['mean_downward_hand_force_n'])}</td><td>{coverage}</td><td>{c['primary_drift_mm']:.2f} / {c['support_drift_mm']:.2f}</td><td>{c['minimum_clearance_mm']:.2f}</td><td>{c['continuous_metrics']['release']['accepted']}</td></tr>"
        )
    parts.append(
        "</table><p>Force means use table-supported samples in the declared force window. Peak includes touchdown. Loaded coverage counts active table contact above 0.1 N from first loaded touchdown to the end of the force window. Lower conditional mean force is not sufficient if coverage worsens. Small variations are not automatically unwanted oscillations; tracking error is measured against the exact last kinematic command, whose age is at most one control period.</p>"
    )
    for c in cases:
        parts.append(
            "<h2>"
            + html.escape(c["label"])
            + "</h2><p>Rejected: "
            + html.escape(", ".join(c["rejected_for"]))
            + "</p><table><tr><th>Window / wrist</th><th>RMS actual−command, mm</th><th>Planned variation RMS, mm</th><th>Actual variation RMS, mm</th><th>Tracking variation RMS, mm</th></tr>"
        )
        for window, roles in c["windows"].items():
            for role, s in roles.items():
                parts.append(
                    f"<tr><td>{html.escape(window + ' / ' + role)}</td><td>{s['tracking_error_m']['rms_mm']:.3f}</td><td>{s['sampled_variation']['planned']['rms_position_variation_mm']:.3f}</td><td>{s['sampled_variation']['actual']['rms_position_variation_mm']:.3f}</td><td>{s['tracking_error_m']['variation']['rms_position_variation_mm']:.3f}</td></tr>"
                )
        parts.append("</table>")
    parts.append(
        '<p><a href="summary.json">All metrics and force samples</a>. Saved 50 Hz samples cannot resolve movement above 25 Hz. Short-timescale variation removes a 0.3 s cubic trend; intended path changes remain distinct from error.</p>'
    )
    (out / "review.html").write_text("\n".join(parts))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.worker:
        run(config)
    else:
        from real2sim2real.bounded_process import run as bounded

        bounded(
            [
                sys.executable,
                "-m",
                __spec__.name,
                "--config",
                str(args.config.resolve()),
                "--worker",
            ],
            timeout=min(config.get("deadline_s", 300), 300),
            env={
                **os.environ,
                "CUDA_VISIBLE_DEVICES": "",
                "OPENBLAS_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
            },
        )
