"""Local object-pose metrics against an immutable grasp-aligned pilot reference.

These diagnostics reuse FlashCHORD metric functions; they are not challenge
scores against held-out ground truth. Motion-only and complete-hold cohorts are
reported separately so stationary padding cannot hide transient error.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from real2sim2real.artifacts import digest, write_json
from flash_chord.evaluation import metrics


def score_poses(actual, reference, vertices):
    """SE(3) rows use xyzw quaternions, metres, one world and one object body."""
    actual, reference = np.asarray(actual), np.asarray(reference)
    if (
        actual.ndim != 2
        or actual.shape[1] != 7
        or actual.shape != reference.shape
        or not len(actual)
    ):
        raise ValueError("Nonempty aligned object pose rows required")
    a, r = actual[:, None, None], reference[:, None]
    add = metrics.object_add(a, r, np.asarray(vertices)[None])
    return dict(
        frames=len(actual),
        mppe_cm=metrics.object_mppe_cm(a, r),
        add_auc=metrics.add_auc(add)[0],
        mean_add_m=float(add.mean()),
        peak_translation_error_m=float(
            np.linalg.norm(actual[:, :3] - reference[:, :3], axis=1).max()
        ),
    )


def evaluate(directory):
    from .object_grasp_pilot import load_scene

    directory = Path(directory)
    path = directory / "result.json"
    report = json.loads(path.read_text())
    for source, sha in report["sources"].items():
        if digest(Path(source)) != sha:
            raise ValueError("Replay source changed")
    if digest(directory / "states.npz") != report["states_sha256"]:
        raise ValueError("Saved replay changed")
    audit = json.loads(Path(report["config"]["audit"]).read_text())
    for source, sha in audit["sources"].items():
        if digest(Path(source)) != sha:
            raise ValueError("Audit reference source changed")
    parent_path = Path(audit["config"]["trial"])
    parent = json.loads(parent_path.read_text())
    from .reference_extension import effective_parent

    parent = effective_parent(parent, audit)
    settings = parent["configuration"]["reference_guidance"]
    motion = Path(settings["path"])
    if digest(motion) != settings["sha256"]:
        raise ValueError("Demonstrated motion reference changed")
    with np.load(motion, allow_pickle=False) as data:
        duration = float(np.ptp(data["timestamps_s"]))
        frame_ids = data["frame_ids"].tolist()
    if len(report["records"]) != len(parent["records"]):
        raise ValueError(
            "Require a complete matched pilot; do not score a truncated success"
        )
    _, _, mesh = load_scene(parent["configuration"])
    indices = np.random.default_rng(0).choice(
        len(mesh.vertices), min(500, len(mesh.vertices)), replace=False
    )
    vertices = np.asarray(mesh.vertices)[indices]
    achieved, references, moving = [], [], []
    for actual, reference in zip(report["records"], parent["records"]):
        if (
            abs(actual["time_s"] - reference["time_s"]) > 0.003
            or actual["phase"] != reference["phase"]
        ):
            raise ValueError("Saved record clock or phase mismatch")
        if reference["phase"] != "hold":
            continue
        achieved.append(
            np.r_[
                actual["object_position_m"],
                Rotation.from_matrix(actual["object_rotation"]).as_quat(),
            ]
        )
        references.append(
            np.r_[
                reference["reference_position_m"],
                Rotation.from_matrix(reference["reference_rotation"]).as_quat(),
            ]
        )
        moving.append(0 <= reference["reference_elapsed_s"] <= duration)
    achieved, references, moving = (
        np.asarray(achieved),
        np.asarray(references),
        np.asarray(moving),
    )
    result = dict(
        schema="real2sim2real.arm-reference-evaluation/1",
        official_challenge_score=False,
        reference="reconstructed object motion, aligned to achieved source grasp",
        clock="source record pairing; maximum allowed clock difference 3 ms; no temporal realignment",
        reference_frame_ids=frame_ids,
        reference_duration_s=duration,
        complete_hold=score_poses(achieved, references, vertices),
        moving_reference=score_poses(achieved[moving], references[moving], vertices),
        mesh_vertex_indices=indices.tolist(),
        sources={
            str(path): digest(path),
            str(parent_path): digest(parent_path),
            str(motion): digest(motion),
        },
        metric_implementation_sha256=digest(Path(metrics.__file__)),
        implementation_sha256=digest(Path(__file__)),
    )
    output = directory / "reference-metrics.json"
    if output.exists():
        raise ValueError("Reference evaluation already exists")
    write_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    evaluate(args.directory)


if __name__ == "__main__":
    main()
