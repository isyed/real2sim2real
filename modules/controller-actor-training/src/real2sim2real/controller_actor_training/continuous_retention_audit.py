"""Read-only closure, early-lift and carry diagnostics on a continuous clock."""

import argparse
import html
import json
from pathlib import Path

import numpy as np

from real2sim2real.artifacts import digest, write_json
from .acquisition_compare import relative_object, pose_difference
from .contact_patch_motion import summarize


def describe(source):
    rows = source["records"]
    protocol = source["continuous_protocol"]
    times = np.array([r["time_s"] for r in rows])
    if not len(times) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Finite strictly increasing continuous record clock required")
    close = next(r for r in rows if r["phase"] == "close")
    anchor = next(r for r in rows if r["time_s"] >= protocol["acquisition_end_s"])
    phases = dict(
        closure=[r for r in rows if r["phase"] == "close"],
        early_lift=[r for r in rows if r["phase"] == "lift"],
        suspended=[r for r in rows if r["phase"] == "hold"],
    )
    windows = {}
    for name, selected in phases.items():
        if not selected:
            raise ValueError("Closure, early lift and suspended observations required")
        pads = {}
        for label in selected[0]["contact_patch_motion"]["pads"]:
            points = [r["contact_patch_motion"]["pads"][label] for r in selected]
            servos = [r["contact_force_feedback"][label] for r in selected]
            pads[label] = dict(
                loaded_fraction=float(np.mean([p["loaded"] for p in points])),
                mean_force_n=float(np.mean([p["force_n"] for p in points])),
                mean_target_n=float(np.mean([s["target_n"] for s in servos])),
                final_normal_gap_m=servos[-1]["normal_gap_m"],
                final_tangent_gap_m=servos[-1]["tangent_gap_m"],
                maximum_offset_norm_rad=max(
                    float(np.linalg.norm(s["joint_offset_rad"])) for s in servos
                ),
                newly_frozen_steps=servos[-1]["frozen_steps"]
                - servos[0]["frozen_steps"],
            )
        windows[name] = dict(
            first_s=selected[0]["time_s"],
            last_s=selected[-1]["time_s"],
            pads=pads,
            wrist_object_offsets_at_end={
                role: dict(
                    translation_m=relative_object(selected[-1], role)[0].tolist(),
                    change_from_close=pose_difference(
                        relative_object(selected[-1], role),
                        relative_object(close, role),
                    ),
                    change_from_acquisition=pose_difference(
                        relative_object(selected[-1], role),
                        relative_object(anchor, role),
                    ),
                )
                for role in ("primary", "secondary")
            },
            maximum_wrist_tracking={
                role: {
                    key: max(r["wrist_tracking"][role][key] for r in selected)
                    for key in ("position_error_m", "rotation_error_deg")
                }
                for role in ("left", "right")
            },
            patch_motion=summarize(selected, source["config"]["contact_patch_motion"]),
        )
    return dict(
        windows=windows,
        acquisition_anchor_s=anchor["time_s"],
        source_success=source["success"],
        source_rejected_for=source["rejected_for"],
        diagnostic_only=True,
        acceptance_changed=False,
    )


def run(paths, output):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    reports, cards = [], []
    for path in paths:
        path = Path(path).resolve()
        source = json.loads(path.read_text())
        for name, key in (
            ("scene.xml", "scene_sha256"),
            ("states.npz", "states_sha256"),
        ):
            if digest(path.with_name(name)) != source[key]:
                raise ValueError("Saved diagnostic artifacts changed")
        report = dict(source=str(path), source_sha256=digest(path), **describe(source))
        reports.append(report)
        cards.append(
            "<h2>"
            + html.escape(path.parent.name)
            + "</h2><p>Original verdict: "
            + html.escape(str(source["rejected_for"]))
            + "</p>"
        )
        for name, window in report["windows"].items():
            cards.append(
                "<h3>"
                + name
                + "</h3><table><tr><th>Pad</th><th>Loaded %</th><th>Mean force / target N</th><th>Final normal / tangent gap mm</th></tr>"
            )
            for label, p in window["pads"].items():
                cards.append(
                    f"<tr><td>{html.escape(label)}</td><td>{100 * p['loaded_fraction']:.1f}</td><td>{p['mean_force_n']:.2f} / {p['mean_target_n']:.2f}</td><td>{1000 * p['final_normal_gap_m']:.2f} / {1000 * p['final_tangent_gap_m']:.2f}</td></tr>"
                )
            cards.append(
                "</table><pre>"
                + html.escape(
                    json.dumps(window["wrist_object_offsets_at_end"], indent=2)
                )
                + "</pre>"
            )
    write_json(
        out / "audit.json",
        dict(
            schema="real2sim2real.continuous-retention-audit/1",
            runs=reports,
            implementation_sha256=digest(Path(__file__)),
        ),
    )
    (out / "review.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Continuous grasp acquisition audit</title><style>body{font-family:system-ui;max-width:1000px;margin:auto}td,th{padding:6px;text-align:left}</style><h1>Closure and early-lift seating</h1><p>Diagnostic windows retain original verdicts. Patch forces use coherent saved-state solves; pressure targets precede the recorded state by one physics step. Tangent gaps are distances from planned pad targets, not independent slip ground truth.</p>'
        + "".join(cards)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.results, args.output)


if __name__ == "__main__":
    main()
