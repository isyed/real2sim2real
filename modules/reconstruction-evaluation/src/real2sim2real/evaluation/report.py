"""Local, inspectable reconstruction report; plotting dependencies are optional."""

import argparse
import html
import json
import os
from pathlib import Path

import numpy as np


def create_report(run, scores, output, *, comparison=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run, scores, output = map(lambda p: Path(p).resolve(), (run, scores, output))
    output.mkdir(parents=True, exist_ok=True)
    if (output / "review.html").exists():
        raise FileExistsError("Keep prior review; select a new report directory")
    metrics = json.loads((scores / "scores.json").read_text())
    errors = np.load(scores / "errors.npz")
    figure, axes = plt.subplots(
        3, 1, figsize=(10, 8), sharex=True, constrained_layout=True
    )
    inputs = [("Estimated camera", errors)]
    if comparison:
        inputs.append(
            ("Fixed-camera diagnostic", np.load(Path(comparison) / "errors.npz"))
        )
    for label, arrays in inputs:
        times = arrays["timestamps_s"][arrays["frame_mask"]]
        axes[0].plot(times, arrays["position_error_m"] * 100, label=label)
        axes[1].plot(times, np.rad2deg(arrays["orientation_error_rad"]), label=label)
        axes[2].plot(times, arrays["add_m"] * 100, label=label)
    for axis, label in zip(
        axes, ["Position error (cm)", "Orientation error (degrees)", "ADD (cm)"]
    ):
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
    axes[0].legend()
    axes[-1].set_xlabel("Capture time (seconds)")
    figure.suptitle("Reconstruction versus reference — one initial rigid alignment")
    figure.savefig(output / "errors.svg")
    figure.savefig(output / "errors.png", dpi=150)
    plt.close(figure)
    labels = {
        "coverage": "Coverage",
        "mean_position_cm": "Mean position (cm)",
        "mean_orientation_deg": "Mean orientation (degrees)",
        "mppe_cm": "MPPE (cm)",
        "add_cm": "ADD (cm)",
        "auc_missing_as_failure": "AUC (missing = failure)",
        "mesh_chamfer_cm": "Registered mesh Chamfer (cm)",
        "mesh_registration_scale": "Mesh registration scale",
        "sp_sr": "SP threshold passed",
        "mp_sr": "MP threshold passed",
    }
    rows = "".join(
        f"<tr><td>{label}</td><td>{html.escape(str(metrics[key]))}</td></tr>"
        for key, label in labels.items()
    )
    timeline = json.loads((run / "timeline.json").read_text())
    frames = []
    for i, (fid, time) in enumerate(
        zip(timeline["frame_ids"], timeline["timestamps_s"])
    ):
        path = run / "overlays" / f"{i:06d}.jpg"
        if path.is_file():
            frames.append(
                dict(id=int(fid), time=float(time), path=os.path.relpath(path, output))
            )
    payload = json.dumps(frames).replace("<", "\\u003c")
    page = """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Reconstruction review</title><style>
body{font:16px system-ui;max-width:1050px;margin:30px auto;padding:0 18px;color:#172330}
table{border-collapse:collapse}td{padding:7px 20px;border-bottom:1px solid #ddd}
img{max-width:100%}input{width:100%}small{display:block;margin:10px 0}
</style><h1>Reconstruction review</h1>
<p>Local single-object evaluation. This measures reconstruction, not robot policy
execution. Reference geometry is used only for evaluation. World alignment occurs
once; trajectory translations are never rescaled. Mesh registration is approximate
and does not claim bitwise parity with an official scorer.</p>
<table>ROWS</table><h2>Errors over capture time</h2><img src="errors.svg" alt="Position, orientation and ADD errors over time">
<h2>Rendered object overlay</h2><input id="frame" type="range" min="0" value="0" aria-label="Frame">
<small id="label"></small><img id="overlay" alt="Rendered predicted object over the input image">
<script>const frames=PAYLOAD;const slider=document.getElementById('frame');
slider.max=Math.max(0,frames.length-1);function show(){const f=frames[Number(slider.value)];
if(!f)return;document.getElementById('label').textContent=`Frame ${f.id} — ${f.time.toFixed(3)} s`;
document.getElementById('overlay').src=f.path;}slider.oninput=show;show();</script></html>"""
    (output / "review.html").write_text(
        page.replace("ROWS", rows).replace("PAYLOAD", payload)
    )
    return output / "review.html"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "scores", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument(
        "--comparison", help="Optional score directory for a diagnostic ablation"
    )
    args = parser.parse_args()
    print(create_report(args.run, args.scores, args.output, comparison=args.comparison))


if __name__ == "__main__":
    main()
