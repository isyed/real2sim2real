"""Bounded matched control experiment using an inferred motion reference."""

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

from real2sim2real.artifacts import digest, write_json
from .guided_motion import MotionReference


def fidelity(records, reference, start, anchor):
    errors, angles = [], []
    for row in records:
        if row["time_s"] < start or row.get("phase", "hold") != "hold":
            continue
        p, r = reference.sample(row["time_s"] - start)
        errors.append(
            float(
                np.linalg.norm(np.asarray(row["object_position_m"]) - anchor[:3, 3] - p)
            )
        )
        angles.append(
            float(
                np.rad2deg(
                    Rotation.from_matrix(
                        np.asarray(row["object_rotation"]) @ (r @ anchor[:3, :3]).T
                    ).magnitude()
                )
            )
        )
    return dict(
        samples=len(errors),
        maximum_position_error_m=max(errors, default=None),
        mean_position_error_m=float(np.mean(errors)) if errors else None,
        maximum_rotation_error_deg=max(angles, default=None),
    )


def run(config):
    from .object_grasp_pilot import load_plan, trial, verify_frozen_controller

    for path, sha in config["sources"].items():
        if digest(Path(path)) != sha:
            raise ValueError("Pinned experiment source changed")
    parent = json.loads(Path(config["parent_trial"]).read_text())
    if (
        not parent["success"]
        or digest(Path(config["parent_trial"]).with_name("states.npz"))
        != parent["states_sha256"]
    ):
        raise ValueError("Pinned successful grasp and states required")
    settings = json.loads(Path(config["scene_config"]).read_text())
    settings["rollout"]["hold_s"] = config["hold_s"]
    verify_frozen_controller(settings, parent)
    if parent["candidate"] not in load_plan(settings)["candidates"]:
        raise ValueError("Parent grasp not in current plan")
    out = Path(config["output"])
    out.mkdir(parents=True, exist_ok=False)
    with np.load(config["object_reference"]) as z:
        select = (z["frame_ids"] >= config["first_frame"]) & (
            z["frame_ids"] <= config["last_frame"]
        )
        reference = MotionReference(
            z["timestamps_s"][select],
            z["world_from_object"][select],
            z["frame_ids"][select],
        )
    np.savez_compressed(
        out / "reference.npz",
        timestamps_s=reference.times,
        world_from_object=reference.poses,
        frame_ids=reference.frame_ids,
    )
    wrist = dict(available=False, reason="No independent wrist observations supplied")
    if config.get("hand_hypotheses"):
        with np.load(config["hand_hypotheses"]) as z:
            if not np.array_equal(
                z["frame_ids"], reference.frame_ids
            ) or not np.allclose(z["timestamps_s"], reference.times, rtol=0, atol=1e-8):
                raise ValueError("Hand/object timelines differ")
            support = (
                z["sequence_admitted"]
                & (z["effective_observation_weights"][:, :, 0] > 0)
                & ~z["observation_identity_ambiguous"][:, :, 0]
            )
            wrist = dict(
                available=bool(support.all()),
                observed_admitted_frames_per_hand=support.sum(0).tolist(),
                reason="Fit admission alone does not establish observed wrist identity; partial or absent evidence is not filled",
            )
            if config.get("detector_landmarks"):
                with np.load(config["detector_landmarks"]) as detector:
                    if not np.array_equal(
                        detector["frame_ids"], reference.frame_ids
                    ) or not np.allclose(
                        detector["timestamps_s"], reference.times, rtol=0, atol=1e-8
                    ):
                        raise ValueError("Detector timeline differs")
                    k = np.asarray(config["detector_intrinsics"])
                    camera = z["camera_joints"][:, :, 0]
                    uvz = camera @ k.T
                    error = np.linalg.norm(
                        uvz[:, :, :2] / uvz[:, :, 2:]
                        - detector["keypoints_px"][:, :, 0],
                        axis=-1,
                    )
                    observed = (
                        detector["accepted"]
                        & np.isfinite(error)
                        & (camera[:, :, 2] > 0)
                    )
                    wrist["detector_observed_frames_per_hand"] = observed.sum(
                        0
                    ).tolist()
                    wrist["detector_wrist_error_median_p90_px"] = [
                        np.quantile(error[observed[:, h], h], [0.5, 0.9]).tolist()
                        if observed[:, h].any()
                        else None
                        for h in range(2)
                    ]
                    wrist["detector_is_independent_ground_truth"] = False
                    wrist["reason"] = (
                        "Point tracks do not observe wrists. Detector wrists are partial, share upstream fitting provenance and do not certify metric targets; human approach is also not phase-aligned to this post-grasp transfer."
                    )
    write_json(
        out / "protocol.json",
        dict(
            config=config,
            bounds=reference.bounds(),
            wrist_evidence=wrist,
            variants=["baseline", "object-guided"],
            wrist_variant_run=False,
            reference_scope="Capture-world increments after the retained staged grasp; original capture start is not reproduced",
            physical_gates_changed=False,
            ground_truth_used=False,
        ),
    )
    results = []
    for name in ["baseline", "object-guided"]:
        current = deepcopy(settings)
        if name == "object-guided":
            current["reference_guidance"] = dict(
                path=str(out / "reference.npz"),
                sha256=digest(out / "reference.npz"),
                limits=config["motion_limits"],
                mode="world_increment_after_grasp",
            )
        result = trial(current, parent["candidate"], out / name)
        rows = [r for r in result["records"] if r["phase"] == "hold"]
        if rows:
            anchor = np.eye(4)
            before = [r for r in result["records"] if r["time_s"] < rows[0]["time_s"]]
            entry = before[-1] if before else rows[0]
            anchor[:3, 3] = entry["object_position_m"]
            anchor[:3, :3] = entry["object_rotation"]
            start = entry["time_s"]
            if name == "object-guided":
                anchor = np.array(result["reference_anchor_world_from_object"])
                start = result["reference_start_time_s"]
            score = fidelity(result["records"], reference, start, anchor)
        else:
            score = dict(
                samples=0,
                maximum_position_error_m=None,
                mean_position_error_m=None,
                maximum_rotation_error_deg=None,
            )
        complete = (
            result["simulation_final_time_s"]
            >= result["simulation_requested_duration_s"] - 1e-6
        )
        passed = (
            complete
            and score["samples"] > 0
            and score["maximum_position_error_m"]
            <= config["fidelity_limits"]["position_m"]
            and score["maximum_rotation_error_deg"]
            <= config["fidelity_limits"]["rotation_deg"]
        )
        row = dict(
            name=name,
            physical_success=result["success"],
            complete=complete,
            physical_rejections=result["rejected_for"],
            fidelity=score,
            reference_fidelity_pass=passed,
            combined_pass=result["success"] and passed,
            states_sha256=result["states_sha256"],
        )
        if name == "baseline" and config.get("baseline_regression_states_sha256"):
            row["baseline_states_identical"] = (
                result["states_sha256"] == config["baseline_regression_states_sha256"]
            )
            if not row["baseline_states_identical"]:
                raise ValueError(
                    "Unchanged baseline failed deterministic state regression"
                )
        results.append(row)
        print(name, row, flush=True)
    report = dict(
        schema="real2sim2real.guided-motion-pilot/1",
        config=config,
        trials=results,
        wrist_evidence=wrist,
        human_finger_imitation_validated=False,
        control_ready=False,
        decision="nominal_motion_transfer_only"
        if results[-1]["combined_pass"]
        else "reference_transfer_not_admitted",
        implementation_sha256={
            str(Path(p)): digest(Path(p))
            for p in [
                __file__,
                str(Path(__file__).with_name("guided_motion.py")),
                str(Path(__file__).with_name("object_grasp_pilot.py")),
            ]
        },
    )
    write_json(out / "evaluation.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if not args.worker:
        from real2sim2real.bounded_process import run as bounded

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
            env={
                **os.environ,
                "CUDA_VISIBLE_DEVICES": "",
                "OPENBLAS_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
            },
        )
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("CPU pilot requires CUDA hidden")
    run(config)


if __name__ == "__main__":
    main()
