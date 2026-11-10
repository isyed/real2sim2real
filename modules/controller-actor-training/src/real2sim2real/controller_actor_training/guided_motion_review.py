"""Review saved motion-transfer trials without rerunning physics."""

import argparse
import html
import json
import os
from pathlib import Path
import sys

from real2sim2real.artifacts import digest, write_json


def review(evaluation, output):
    from .object_grasp_review import review_trials

    evaluation, output = Path(evaluation), Path(output)
    audit = json.loads(evaluation.read_text())
    config = json.loads(Path(audit["config"]["scene_config"]).read_text())
    if (
        digest(Path(audit["config"]["scene_config"]))
        != audit["config"]["sources"][audit["config"]["scene_config"]]
    ):
        raise ValueError("Reviewed scene configuration changed")
    trials = [evaluation.parent / row["name"] for row in audit["trials"]]
    for row, trial in zip(audit["trials"], trials):
        if digest(trial / "states.npz") != row["states_sha256"]:
            raise ValueError("Reviewed states changed")
    review_trials(
        config,
        trials,
        output,
        title="Reconstructed motion: bounded transfer after grasp",
    )
    table = []

    def metric(value, scale=1):
        return "unavailable" if value is None else f"{value * scale:.1f}"

    for row, trial in zip(audit["trials"], trials):
        detail = json.loads((trial / "result.json").read_text())
        f = row["fidelity"]
        values = [
            row["name"],
            row["physical_success"],
            row["reference_fidelity_pass"],
            metric(f["mean_position_error_m"], 1000),
            metric(f["maximum_position_error_m"], 1000),
            metric(f["maximum_rotation_error_deg"]),
            metric(detail["hold_slip_m"], 1000),
        ]
        table.append(
            "<tr>"
            + "".join(f"<td>{html.escape(str(v))}</td>" for v in values)
            + "</tr>"
        )
    intro = (
        "<p><strong>This is a post-grasp motion-transfer diagnostic, not original-capture replay or human finger imitation.</strong> "
        "Both runs retain the same staged grasp, object/table geometry, material assumptions, force limits and physical gates. "
        "The guided variant applies capture-world object-motion increments after the second hand seats, preserving capture timing. "
        "Its start is anchored to the achieved simulated grasp; the original floating reconstruction and changed start are not corrected by this trial. "
        "Exact simulator feedback is privileged. No learned policy, arm or hardware execution is validated.</p>"
        f"<p>Physical acceptance and motion fidelity are separate. The diagnostic fidelity limits are {1000 * audit['config']['fidelity_limits']['position_m']:g} mm maximum position error and {audit['config']['fidelity_limits']['rotation_deg']:g} degrees orientation error; "
        "the existing physical wrist-relative slip limit remains 10 mm. Low motion error cannot erase a failed grasp gate.</p>"
        "<table><tr><th>Variant</th><th>Physical pass</th><th>Fidelity pass</th><th>Mean error mm</th><th>Max error mm</th><th>Max angle error degrees</th><th>Primary slip mm</th></tr>"
        + "".join(table)
        + "</table><p>Human wrist guidance was not executed. "
        + html.escape(audit["wrist_evidence"]["reason"])
        + "</p>"
        '<p><a href="evaluation.json">Complete comparison</a></p>'
    )
    p = output / "review.html"
    content = p.read_text()
    content = content.replace("</h1>", "</h1>" + intro, 1)
    p.write_text(content)
    write_json(output / "evaluation.json", audit)
    receipt = json.loads((output / "review.json").read_text())
    receipt.update(
        evaluation_sha256=digest(evaluation),
        implementation_sha256=digest(Path(__file__)),
    )
    write_json(output / "review.json", receipt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if not args.worker:
        from real2sim2real.bounded_process import run

        return run(
            [
                sys.executable,
                "-m",
                __spec__.name,
                "--evaluation",
                str(args.evaluation.resolve()),
                "--output",
                str(args.output.resolve()),
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
    review(args.evaluation, args.output)


if __name__ == "__main__":
    main()
