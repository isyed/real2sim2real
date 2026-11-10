"""Review frozen retention comparisons and saved physics without rerunning it."""

import argparse
import html
import json
import os
from pathlib import Path
import sys

from real2sim2real.artifacts import digest, write_json


def review(comparison, output):
    from .object_grasp_review import review_trials

    comparison, output = Path(comparison), Path(output)
    audit = json.loads(comparison.read_text())
    parent_path = Path(audit["config"]["parent_trial"])
    if digest(parent_path) != audit["config"]["sources"][str(parent_path)]:
        raise ValueError("Reviewed parent changed")
    config = json.loads(parent_path.read_text())["configuration"]
    selected = audit["selection"]["selected"]
    rendered, seen, aliases = [], {}, {}
    table = []
    for row in audit["trials"]:
        name = row["directory"]
        trial = comparison.parent / name
        if digest(trial / "states.npz") != row["states_sha256"]:
            raise ValueError("Reviewed states changed")
        f = row["fidelity"]
        values = [
            name,
            row["success"],
            row["reference_pass"],
            f"{1000 * row['hold_slip_m']:.2f}",
            "unavailable"
            if f is None
            else f"{1000 * f['maximum_position_error_m']:.2f}",
            "unavailable" if f is None else f"{f['maximum_rotation_error_deg']:.2f}",
        ]
        table.append(
            "<tr>"
            + "".join(f"<td>{html.escape(str(v))}</td>" for v in values)
            + "</tr>"
        )
        if name in ("baseline", selected) or name.startswith("validation-"):
            if row["states_sha256"] in seen:
                aliases[name] = seen[row["states_sha256"]]
            else:
                rendered.append(trial)
                seen[row["states_sha256"]] = name
    review_trials(
        config, rendered, output, title="Demonstrated object motion: contact retention"
    )
    intro = (
        "<p><strong>Decision: " + html.escape(audit["decision"]) + ".</strong> "
        "The demonstrated object-motion increments, capture timing and achieved-grasp reference anchor are fixed. "
        "A bounded primary-wrist translation correction regulates the object-to-wrist vector only while opposed contacts exist and reference error is small. "
        "It uses exact simulator poses and forces. Physical gates are unchanged; this is not a learned policy, human finger imitation or original-start replay.</p>"
        "<p>Wrist-relative displacement measures retention, not surface-point slip. A wrist can follow a drifting object, so contact forces and reference fidelity must also be checked. "
        "Longer duration and altered friction are frozen validation cases; nominal success does not erase their failures. "
        "The world-motion figures below include the intended trajectory and are not tracking-error scores.</p>"
        "<table><tr><th>Trial</th><th>Physical pass</th><th>Fidelity pass</th><th>Primary slip mm</th><th>Max reference error mm</th><th>Max angle error degrees</th></tr>"
        + "".join(table)
        + "</table>"
        + "<p>Videos show the baseline, selected candidate and validation states. Other development trials remain in the table. "
        + " ".join(
            html.escape(f"{k} has identical saved states to {v}; they share a video.")
            for k, v in aliases.items()
        )
        + '</p><p><a href="comparison.json">Complete comparison and protocol</a></p>'
    )
    page = output / "review.html"
    page.write_text(page.read_text().replace("</h1>", "</h1>" + intro, 1))
    write_json(output / "comparison.json", audit)
    receipt = json.loads((output / "review.json").read_text())
    receipt.update(
        comparison_sha256=digest(comparison),
        identical_state_aliases=aliases,
        implementation_sha256=digest(Path(__file__)),
    )
    write_json(output / "review.json", receipt)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--comparison", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--worker", action="store_true")
    a = p.parse_args()
    if not a.worker:
        from real2sim2real.bounded_process import run

        return run(
            [
                sys.executable,
                "-m",
                __spec__.name,
                "--comparison",
                str(a.comparison.resolve()),
                "--output",
                str(a.output.resolve()),
                "--worker",
            ],
            timeout=600,
            env={
                **os.environ,
                "CUDA_VISIBLE_DEVICES": "",
                "OPENBLAS_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
            },
        )
    review(a.comparison, a.output)


if __name__ == "__main__":
    main()
