"""Read-only, pinned contact/retention feedback for bounded policy revision.

Only named development runs are exported. No new physics, changed anchors or
inferred acceptance are introduced; sparse diagnostics retain their limitations.
"""

import argparse
import json
from copy import deepcopy
from pathlib import Path

import numpy as np

from real2sim2real.artifacts import digest, write_json

from .acquisition_compare import relative_object
from .continuous_retention_audit import load_readiness
from .paired_acquisition_audit import pinned_json


def statistics(values):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite feedback observations")
    return {
        "samples": len(values),
        "mean": float(np.mean(values)) if len(values) else None,
        "maximum": float(np.max(values)) if len(values) else None,
    }


def retention_series(source):
    """Use the evaluator's acquisition-to-pre-release cohort, never re-anchor."""
    protocol = source["continuous_protocol"]
    rows = source["records"]
    times = np.asarray([r["time_s"] for r in rows])
    if not len(times) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Finite strictly increasing saved observation times required")
    selected = [
        r
        for r in rows
        if protocol["acquisition_end_s"]
        <= r["time_s"]
        < protocol["finger_release"]["start_s"]
    ]
    if not selected:
        raise ValueError("No acquisition-anchored retention observations")
    result = []
    for row in selected:
        result.append(
            {
                "time_s": row["time_s"],
                "phase": row["phase"],
                **{
                    role + "_drift_m": float(
                        np.linalg.norm(
                            relative_object(row, role)[0]
                            - relative_object(selected[0], role)[0]
                        )
                    )
                    for role in ("primary", "secondary")
                },
            }
        )
    return result


def first_breach(rows, key, limit, *, minimum=False):
    for row in rows:
        value = row[key]
        if not np.isfinite(value):
            raise ValueError("Nonfinite gate diagnostic")
        if (value < limit) if minimum else (value > limit):
            return {"time_s": row["time_s"], "phase": row["phase"], "value": value}
    return None


def pad_statistics(rows, label):
    points = [r["contact_patch_motion"]["pads"][label] for r in rows]
    servos = [r["contact_force_feedback"][label] for r in rows]
    loaded = [p for p in points if p["loaded"]]
    return {
        "loaded_sample_fraction": float(np.mean([p["loaded"] for p in points])),
        "coherent_force_n": statistics([p["force_n"] for p in points]),
        "preceding_controller_target_n": statistics([s["target_n"] for s in servos]),
        "sliding_fraction_of_loaded_samples": float(
            np.mean([p["sliding"] for p in loaded])
        )
        if loaded
        else None,
        "loaded_tangent_speed_m_s": statistics(
            [p["tangent_speed_m_s"] for p in loaded]
        ),
        "loaded_linear_friction_ratio": statistics(
            [
                p["maximum_linear_friction_ratio"]
                for p in loaded
                if p.get("maximum_linear_friction_ratio") is not None
            ]
        ),
        "maximum_servo_joint_offset_norm_rad": max(
            float(np.linalg.norm(s["joint_offset_rad"])) for s in servos
        ),
        "final_planned_normal_gap_m": servos[-1]["normal_gap_m"],
        "final_planned_tangent_gap_m": servos[-1]["tangent_gap_m"],
        "loaded_mean_object_local_centroid_m": np.mean(
            [p["object_local_centroid_m"] for p in loaded], axis=0
        ).tolist()
        if loaded
        else None,
    }


def summarize(source, table_z_m, limits):
    rows = source["records"]
    retained = retention_series(source)
    for role, key in (
        ("primary", "hold_slip_m"),
        ("secondary", "support_wrist_drift_m"),
    ):
        if not np.isclose(
            max(r[role + "_drift_m"] for r in retained), source[key], rtol=0, atol=1e-12
        ):
            raise ValueError("Retention diagnostics do not reproduce the source metric")
    windows = {}
    for phase in ("close", "lift", "hold", "return"):
        selected = [r for r in rows if r["phase"] == phase]
        if not selected:
            raise ValueError("Complete closure/lift/hold/return observations required")
        labels = set(selected[0]["contact_patch_motion"]["pads"])
        if any(set(r["contact_patch_motion"]["pads"]) != labels for r in selected):
            raise ValueError("Pad identity changed within a feedback window")
        windows[phase] = {
            "first_s": selected[0]["time_s"],
            "last_s": selected[-1]["time_s"],
            "samples": len(selected),
            "pads": {
                label: pad_statistics(selected, label) for label in sorted(labels)
            },
            "controller_reported_upward_force_n": {
                hand: statistics(
                    [r["payload_pressure"]["upward_force_n"][i] for r in selected]
                )
                for i, hand in enumerate(("primary", "secondary"))
            },
            "maximum_commanded_wrist_tracking_error_m": {
                hand: max(
                    r["commanded_wrist_tracking"][hand]["position_error_m"]
                    for r in selected
                )
                for hand in ("left", "right")
            },
            "other_object_contact_mean_force_n": {
                name: float(
                    np.mean(
                        [
                            r["contact_patch_motion"]["other_object_contacts_n"].get(
                                name, 0
                            )
                            for r in selected
                        ]
                    )
                )
                for name in sorted(
                    set().union(
                        *(
                            r["contact_patch_motion"]["other_object_contacts_n"]
                            for r in selected
                        )
                    )
                )
            },
        }
    hold = [
        {**r, "clearance_m": r["minimum_object_height_m"] - table_z_m}
        for r in rows
        if r["phase"] == "hold"
    ]
    return {
        "original_success": source["success"],
        "original_rejected_for": source["rejected_for"],
        "parameters_relative_to_frozen_controller": deepcopy(
            source.get("bounded_grasp_correction", {}).get("spec")
        ),
        "retention_anchor_s": retained[0]["time_s"],
        "maximum_saved_sample_gap_s": float(
            np.max(np.diff([r["time_s"] for r in rows]))
        ),
        "first_sampled_gate_breaches": {
            "primary_drift": first_breach(
                retained, "primary_drift_m", limits["drift_m"]
            ),
            "support_drift": first_breach(
                retained, "secondary_drift_m", limits["drift_m"]
            ),
            "suspended_clearance": first_breach(
                hold, "clearance_m", limits["clearance_m"], minimum=True
            ),
            "reference_position": first_breach(
                rows, "reference_position_error_m", limits["reference_position_m"]
            ),
            "reference_rotation": first_breach(
                rows, "reference_rotation_error_deg", limits["reference_rotation_deg"]
            ),
        },
        "metrics": {
            k: source[k]
            for k in (
                "hold_slip_m",
                "support_wrist_drift_m",
                "hold_minimum_clearance_m",
                "max_robot_penetration_m",
                "max_arm_penetration_m",
                "maximum_dynamic_velocity_ratio",
                "minimum_dynamic_joint_margin_rad",
            )
        },
        "release": source["continuous_metrics"]["release"],
        "windows": windows,
        "load_readiness": load_readiness(
            [r for r in rows if r["phase"] in ("close", "lift", "hold")]
        ),
        "retention_events": {
            role: max(retained, key=lambda r: r[role + "_drift_m"])
            for role in ("primary", "secondary")
        },
    }


def sampled_collisions(path, source, arms, penetration_limit_m):
    """Kinematic contact distances at saved states; not unsaved substep peaks."""
    import mujoco

    from .arm_path_audit import joint_neighbors
    from .arm_replay_pilot import subtree_geoms

    model = mujoco.MjModel.from_xml_path(str(path.with_name("scene.xml")))
    data = mujoco.MjData(model)
    arm_bodies = {model.body(n).id for arm in arms for n in arm["collision_bodies"]}
    arm_geoms = {
        g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in arm_bodies
    }
    robot = arm_geoms | set().union(
        *(subtree_geoms(model, model.body(a["root"]).id) for a in arms)
    )
    peaks = {}
    first = None
    with np.load(path.with_name("states.npz"), allow_pickle=False) as states:
        if not np.array_equal(
            states["timestamps_s"], [r["time_s"] for r in source["records"]]
        ):
            raise ValueError("Saved states and contact feedback clocks differ")
        for q, row in zip(states["qpos"], source["records"], strict=True):
            data.qpos[:] = q
            mujoco.mj_fwdPosition(model, data)
            for contact in data.contact:
                pair = {int(contact.geom1), int(contact.geom2)}
                bodies = [
                    int(model.geom_bodyid[g]) for g in (contact.geom1, contact.geom2)
                ]
                if not pair & robot or (
                    pair & arm_geoms and joint_neighbors(model, *bodies)
                ):
                    continue
                depth = max(0.0, -float(contact.dist))
                names = sorted(model.geom(g).name or f"geom-{g}" for g in pair)
                key = " / ".join(names)
                event = {
                    "time_s": row["time_s"],
                    "phase": row["phase"],
                    "penetration_m": depth,
                    "geoms": names,
                }
                if key not in peaks or depth > peaks[key]["penetration_m"]:
                    peaks[key] = event
                if depth > penetration_limit_m and first is None:
                    first = event
    return {
        "method": "read-only saved-qpos kinematic contacts; sampled, not original substep peak timing",
        "first_sampled_robot_limit_breach": first,
        "largest_contact_pairs": sorted(
            peaks.values(), key=lambda r: r["penetration_m"], reverse=True
        )[:8],
    }


def build(config):
    from .passthrough_graph_validate import source_files

    reports = []
    provenance = []
    for entry in config["runs"]:
        source = pinned_json(entry["result"])
        path = Path(entry["result"]["path"])
        pinned = source_files(path)
        audit = pinned_json(
            {
                "path": source["config"]["audit"],
                "sha256": source["config"]["audit_sha256"],
            }
        )
        trial = json.loads(Path(audit["config"]["trial"]).read_text())
        task = json.loads(Path(trial["configuration"]["task"]).read_text())
        limits = {
            "drift_m": 0.01,
            "clearance_m": task["goal"]["minimum_clearance_m"],
            "reference_position_m": 0.02,
            "reference_rotation_deg": 10,
            "robot_penetration_m": trial["configuration"]["planning"][
                "max_penetration_m"
            ],
            "arm_penetration_m": audit["config"]["criteria"]["arm_penetration_m"],
            "joint_margin_rad": audit["config"]["criteria"]["joint_margin_rad"],
        }
        summary = summarize(source, task["table"]["top_z_m"], limits)
        # Paths are provenance, not useful author feedback; omit them from the packet.
        if summary["parameters_relative_to_frozen_controller"] is not None:
            summary["parameters_relative_to_frozen_controller"].pop("baseline", None)
        reports.append(
            {
                "label": entry["label"],
                "limits": limits,
                **summary,
                "sampled_collisions": sampled_collisions(
                    path, source, audit["config"]["arms"], limits["robot_penetration_m"]
                ),
            }
        )
        if pinned != source_files(path):
            raise ValueError("Feedback inputs changed during export")
        provenance.append(
            {"label": entry["label"], "result": entry["result"], "source_files": pinned}
        )
    out = Path(config["output"])
    out.mkdir(parents=True, exist_ok=False)
    packet = {
        "schema": "bounded-policy-feedback/1",
        "runs": reports,
        "notes": [
            "No physics rerun or acceptance change.",
            "Force/sliding telemetry is coherent sampled simulation evidence, not real tactile ground truth.",
            "Pressure targets and load estimates are preceding-step controller observations; coherent pad force is a separate saved-state solve.",
            "Planned tangent gaps and contact-centroid migration are not independent material slip measurements.",
            "A missing sampled breach does not refute an original unsampled substep peak.",
            "All revised offsets remain absolute relative to the original frozen controller; do not stack them on the seed candidate.",
        ],
    }
    write_json(out / "feedback.json", packet)
    write_json(
        out / "provenance.json",
        {
            "runs": provenance,
            "feedback_sha256": digest(out / "feedback.json"),
            "implementation_sha256": digest(Path(__file__)),
        },
    )
    return packet


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    build(json.loads(args.config.read_text()))
