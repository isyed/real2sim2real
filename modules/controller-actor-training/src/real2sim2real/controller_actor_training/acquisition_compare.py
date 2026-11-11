"""Read-only comparison of pinned, state-identical grasp diagnostic replays.

Keeps physical/reference verdicts intact. Sampled contact forces are distinct
from the controller's preceding-step observations and requests.
"""

import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from real2sim2real.artifacts import digest, write_json

PHASES = ("approach", "close", "lift", "hold")


def changed_paths(a, b, prefix=""):
    if isinstance(a, dict) and isinstance(b, dict):
        paths = []
        for key in sorted(a.keys() | b.keys()):
            path = f"{prefix}.{key}" if prefix else key
            if key not in a or key not in b:
                paths.append(path)
            else:
                paths.extend(changed_paths(a[key], b[key], path))
        return paths
    return [] if a == b else [prefix]


def relative_object(row, role):
    """Actual object center/orientation in the measured wrist frame."""
    key = "wrist_" if role == "primary" else "support_wrist_"
    wrist = np.asarray(row[key + "rotation"], float)
    position = np.asarray(row["object_position_m"], float)
    rotation = np.asarray(row["object_rotation"], float)
    return (
        wrist.T @ (position - row[key + "position_m"]),
        wrist.T @ rotation,
    )


def pose_difference(a, b):
    return dict(
        translation_m=float(np.linalg.norm(a[0] - b[0])),
        rotation_deg=float(np.rad2deg(Rotation.from_matrix(a[1] @ b[1].T).magnitude())),
    )


def validate_records(records, minimum_force_n):
    if not records:
        raise ValueError("Nonempty diagnostic records required")
    times = np.array([r["time_s"] for r in records], float)
    phases = [r["phase"] for r in records]
    if (
        not np.isfinite(times).all()
        or np.any(np.diff(times) <= 0)
        or set(phases) != set(PHASES)
        or any(PHASES.index(a) > PHASES.index(b) for a, b in zip(phases, phases[1:]))
    ):
        raise ValueError("Strict native clock and ordered complete phases required")
    labels = set(records[0]["contact_patch_motion"]["pads"])
    if not labels or any(
        label.split(":")[0] not in ("primary", "secondary") for label in labels
    ):
        raise ValueError("Explicit bilateral pad labels required")
    for row in records:
        pads = row["contact_patch_motion"]["pads"]
        if set(pads) != labels or set(row["contact_force_feedback"]) != labels:
            raise ValueError("Identical pad identities required throughout acquisition")
        for label, pad in pads.items():
            servo = row["contact_force_feedback"][label]
            values = [
                pad["force_n"],
                servo["force_n"],
                servo["target_n"],
                *servo["joint_offset_rad"],
            ]
            if not np.isfinite(values).all() or min(values[:3]) < 0:
                raise ValueError(
                    "Finite nonnegative forces and finite offsets required"
                )
            if type(pad["loaded"]) is not bool or pad["loaded"] != bool(
                pad["force_n"] >= minimum_force_n
            ):
                raise ValueError("Contact loading threshold does not match the audit")
            if pad["loaded"]:
                for key in ("object_local_centroid_m", "pad_local_centroid_m"):
                    value = np.asarray(pad[key], float)
                    if value.shape != (3,) or not np.isfinite(value).all():
                        raise ValueError("Finite loaded contact centroids required")
                if (
                    not np.isfinite(pad["tangent_speed_m_s"])
                    or pad["tangent_speed_m_s"] < 0
                ):
                    raise ValueError(
                        "Finite nonnegative sampled sliding speed required"
                    )
    return times, sorted(labels)


def describe(records, settings):
    times, labels = validate_records(records, settings["minimum_force_n"])
    first_close = next(r for r in records if r["phase"] == "close")
    first_hold = next(r for r in records if r["phase"] == "hold")
    phase_rows = {}
    for phase in PHASES:
        selected = [r for r in records if r["phase"] == phase]
        pads = {}
        for label in labels:
            points = [r["contact_patch_motion"]["pads"][label] for r in selected]
            servos = [r["contact_force_feedback"][label] for r in selected]
            active = [
                (p, s)
                for p, s in zip(points, servos)
                if s["updated_steps"] + s["frozen_steps"] > 0
            ]
            loaded = [p for p in points if p["loaded"]]
            sliding, migration = 0.0, 0.0
            for a, b, ra, rb in zip(points, points[1:], selected, selected[1:]):
                dt = rb["time_s"] - ra["time_s"]
                if (
                    a["loaded"]
                    and b["loaded"]
                    and dt <= settings["maximum_sample_gap_s"]
                ):
                    sliding += (
                        dt * (a["tangent_speed_m_s"] + b["tangent_speed_m_s"]) / 2
                    )
                    migration += float(
                        np.linalg.norm(
                            np.asarray(b["object_local_centroid_m"])
                            - a["object_local_centroid_m"]
                        )
                    )
            pads[label] = dict(
                loaded_sample_fraction=len(loaded) / len(points),
                mean_coherent_force_n=float(np.mean([p["force_n"] for p in points])),
                peak_coherent_force_n=max(p["force_n"] for p in points),
                active_control_samples=len(active),
                active_mean_force_deficit_n=float(
                    np.mean([max(0, s["target_n"] - p["force_n"]) for p, s in active])
                )
                if active
                else None,
                sampled_sliding_path_m=sliding,
                object_centroid_migration_m=migration,
                maximum_offset_norm_rad=max(
                    float(np.linalg.norm(s["joint_offset_rad"])) for s in servos
                ),
                final_offset_rad=servos[-1]["joint_offset_rad"],
                final_coherent_force_n=points[-1]["force_n"],
                final_target_force_n=servos[-1]["target_n"],
            )
        other_names = set().union(
            *(r["contact_patch_motion"]["other_object_contacts_n"] for r in selected)
        )
        phase_rows[phase] = dict(
            first_sample_s=selected[0]["time_s"],
            last_sample_s=selected[-1]["time_s"],
            samples=len(selected),
            pads=pads,
            other_contact_mean_force_n={
                name: float(
                    np.mean(
                        [
                            r["contact_patch_motion"]["other_object_contacts_n"].get(
                                name, 0.0
                            )
                            for r in selected
                        ]
                    )
                )
                for name in sorted(other_names)
            },
            final_object_wrist_change_from_first_close={
                role: pose_difference(
                    relative_object(selected[-1], role),
                    relative_object(first_close, role),
                )
                for role in ("primary", "secondary")
            },
            final_object_wrist_change_from_first_hold={
                role: pose_difference(
                    relative_object(selected[-1], role),
                    relative_object(first_hold, role),
                )
                for role in ("primary", "secondary")
            }
            if phase == "hold"
            else None,
        )
    first_loaded = {
        label: next(
            (
                r["time_s"]
                for r in records
                if r["contact_patch_motion"]["pads"][label]["loaded"]
            ),
            None,
        )
        for label in labels
    }
    return dict(
        phases=phase_rows,
        first_loaded_pad_sample_s=first_loaded,
        first_each_hand_has_loaded_pad_sample_s=next(
            (
                r["time_s"]
                for r in records
                if all(
                    any(
                        p["loaded"]
                        for label, p in r["contact_patch_motion"]["pads"].items()
                        if label.startswith(role + ":")
                    )
                    for role in ("primary", "secondary")
                )
            ),
            None,
        ),
        first_no_table_contact_sample_s=next(
            (r["time_s"] for r in records if r["table_contacts"] == 0), None
        ),
        first_supported_load_update_sample_s=next(
            (
                r["time_s"]
                for r in records
                if r["payload_pressure"]["valid_updates"] > 0
            ),
            None,
        ),
        first_hold_sample_s=first_hold["time_s"],
        first_transition_sample_s=next(
            (
                r["time_s"]
                for r in records
                if r.get("payload_posture", {}).get("fraction", 0) > 0
            ),
            None,
        ),
        first_full_transition_sample_s=next(
            (
                r["time_s"]
                for r in records
                if r.get("payload_posture", {}).get("fraction", 0) >= 1
            ),
            None,
        ),
        sample_count=len(times),
    )


def load_audit(path, sha):
    path = Path(path)
    if digest(path) != sha:
        raise ValueError("Pinned acquisition audit changed")
    audit = json.loads(path.read_text())
    source_path = Path(audit["config"]["source_result"])
    replay_path = path.parent / "replay/result.json"
    if (
        digest(source_path) != audit["config"]["source_sha256"]
        or digest(replay_path) != audit["replay_result_sha256"]
    ):
        raise ValueError("Pinned source or replay result changed")
    source, replay = (
        json.loads(source_path.read_text()),
        json.loads(replay_path.read_text()),
    )
    if (
        audit["states_identical"] is not True
        or source["states_sha256"] != replay["states_sha256"]
        or source["states_sha256"] != audit["source_states_sha256"]
        or digest(source_path.with_name("states.npz")) != source["states_sha256"]
        or digest(replay_path.with_name("states.npz")) != replay["states_sha256"]
    ):
        raise ValueError("Byte-identical source/replay state archives required")
    if (
        not np.isfinite(
            [
                source["simulation_final_time_s"],
                source["simulation_requested_duration_s"],
            ]
        ).all()
        or source["simulation_requested_duration_s"] <= 0
        or source["simulation_final_time_s"]
        < source["simulation_requested_duration_s"] - 1e-6
    ):
        raise ValueError("Complete finite source rollout required")
    if source["success"] is not replay["success"]:
        raise ValueError("Diagnostic replay changed the physical verdict")
    return source, replay, audit["config"]["diagnostic_settings"]


def compare(baseline, candidate, settings, allowed_changes):
    from .contact_patch_motion import validate_settings

    validate_settings(settings)
    a, b = baseline["records"], candidate["records"]
    ta, la = validate_records(a, settings["minimum_force_n"])
    tb, lb = validate_records(b, settings["minimum_force_n"])
    if (
        ta.shape != tb.shape
        or not np.allclose(ta, tb, atol=1e-9, rtol=0)
        or la != lb
        or [r["phase"] for r in a] != [r["phase"] for r in b]
    ):
        raise ValueError("Matched native sample clocks, phases and identities required")
    for key in (
        "source_scene_sha256",
        "mass_scale",
        "friction",
        "simulation_timestep_s",
        "simulation_requested_duration_s",
    ):
        if baseline[key] != candidate[key]:
            raise ValueError("Comparison must preserve scene, load, friction and clock")
    configs = [deepcopy(r["configuration"]) for r in (baseline, candidate)]
    for c in configs:
        c.pop("planning_directory", None)
        c.pop("contact_patch_motion", None)
    changed = changed_paths(*configs)
    if not set(changed) <= set(allowed_changes):
        raise ValueError(f"Undeclared configuration changes: {changed}")
    for first, second in (
        (baseline["candidate"], candidate["candidate"]),
        (baseline["candidate"]["support"], candidate["candidate"]["support"]),
    ):
        if first["joint_names"] != second["joint_names"] or any(
            not np.allclose(first[k], second[k], atol=1e-6, rtol=0)
            for k in ("base_position_m", "base_rotation", "targets_m", "q_open")
        ):
            raise ValueError("Fixed wrists, open posture and object targets required")
        prefix = first.get("fixture_prefix", "fixture_")
        fixture_names = [prefix + axis for axis in ("x", "y", "z", "rx", "ry", "rz")]
        if set(fixture_names) & set(first["joint_names"]):
            if not set(fixture_names) <= set(first["joint_names"]):
                raise ValueError("Complete declared Cartesian wrist fixture required")
            owned = [first["joint_names"].index(n) for n in fixture_names]
            for key in ("q_grasp", "q_closed"):
                if not np.allclose(
                    np.asarray(first[key])[owned],
                    np.asarray(second[key])[owned],
                    atol=1e-6,
                    rtol=0,
                ):
                    raise ValueError("Closed/grasp wrist fixture coordinates changed")
    pair_rows = []
    for x, y in zip(a, b):
        pads = {}
        for label in la:
            px, py = [r["contact_patch_motion"]["pads"][label] for r in (x, y)]
            sx, sy = [r["contact_force_feedback"][label] for r in (x, y)]
            if len(sx["joint_offset_rad"]) != len(sy["joint_offset_rad"]):
                raise ValueError("Matching controlled finger coordinates required")
            pads[label] = dict(
                offset_difference_norm_rad=float(
                    np.linalg.norm(
                        np.asarray(sy["joint_offset_rad"]) - sx["joint_offset_rad"]
                    )
                ),
                object_centroid_separation_m=float(
                    np.linalg.norm(
                        np.asarray(py["object_local_centroid_m"])
                        - px["object_local_centroid_m"]
                    )
                )
                if px["loaded"] and py["loaded"]
                else None,
            )
        pair_rows.append(
            dict(
                time_s=x["time_s"],
                phase=x["phase"],
                pads=pads,
                wrist_relative_object_difference={
                    role: pose_difference(
                        relative_object(x, role), relative_object(y, role)
                    )
                    for role in ("primary", "secondary")
                },
            )
        )
    return dict(
        schema="real2sim2real.acquisition-comparison/1",
        baseline=describe(a, settings),
        candidate=describe(b, settings),
        declared_configuration_changes=allowed_changes,
        actual_configuration_changes=changed,
        original_physical_verdicts=dict(
            baseline=baseline["success"], candidate=candidate["success"]
        ),
        sampled_pair_differences=pair_rows,
        diagnostic_only=True,
        acceptance_changed=False,
        source_verdicts_promoted=False,
        force_clock_note="Patch forces are coherent post-step mj_forward samples; pressure requests/offsets and servo force observations precede that state by one physics step. No force-settling certificate is inferred.",
        anchor_note="Relative pose changes use first-close and first-hold diagnostic samples only. They do not replace the source acceptance anchors or drift maxima.",
    )


def review(report, records, output):
    payload = json.dumps(dict(report=report, records=records), allow_nan=False).replace(
        "<", "\\u003c"
    )
    template = Path(__file__).with_name("acquisition_review.html").read_text()
    output.write_text(template.replace("__DATA__", payload))


def run(config):
    if set(config) != {
        "baseline_audit",
        "baseline_sha256",
        "candidate_audit",
        "candidate_sha256",
        "allowed_configuration_changes",
        "output",
        "deadline_s",
    }:
        raise ValueError(
            "Pinned audit pair and explicit changed configuration paths required"
        )
    if not isinstance(config["allowed_configuration_changes"], list) or any(
        not isinstance(p, str) or not p for p in config["allowed_configuration_changes"]
    ):
        raise ValueError("Explicit configuration paths required")
    baseline, a, sa = load_audit(config["baseline_audit"], config["baseline_sha256"])
    candidate, b, sb = load_audit(config["candidate_audit"], config["candidate_sha256"])
    if sa != sb:
        raise ValueError("Identical contact diagnostic thresholds required")
    report = compare(a, b, sa, config["allowed_configuration_changes"])
    report["config"] = config
    report["source_result_sha256"] = {
        name: json.loads(Path(path).read_text())["config"]["source_sha256"]
        for name, path in (
            ("baseline", config["baseline_audit"]),
            ("candidate", config["candidate_audit"]),
        )
    }
    report["replay_result_sha256"] = {
        "baseline": digest(
            Path(config["baseline_audit"]).parent / "replay/result.json"
        ),
        "candidate": digest(
            Path(config["candidate_audit"]).parent / "replay/result.json"
        ),
    }
    report["implementation_sha256"] = digest(Path(__file__))
    report["review_template_sha256"] = digest(
        Path(__file__).with_name("acquisition_review.html")
    )
    out = Path(config["output"])
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "comparison.json", report)
    # The viewer uses coherent contacts, pressure requests and offsets, not models.
    review(
        report, dict(baseline=a["records"], candidate=b["records"]), out / "review.html"
    )
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--worker", action="store_true")
    args = p.parse_args()
    config = json.loads(args.config.read_text())
    if not np.isfinite(config["deadline_s"]) or not 0 < config["deadline_s"] <= 120:
        raise ValueError("Comparison deadline of at most 120 seconds required")
    if args.worker:
        run(config)
    else:
        from real2sim2real.bounded_process import run as bounded
        import sys

        return bounded(
            [
                sys.executable,
                "-m",
                __spec__.name,
                "--config",
                str(args.config.resolve()),
                "--worker",
            ],
            timeout=config["deadline_s"],
        )


if __name__ == "__main__":
    main()
