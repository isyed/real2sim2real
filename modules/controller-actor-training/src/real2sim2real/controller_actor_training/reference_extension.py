"""Extend a frozen grasp's object reference before a fresh arm-path audit."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import numpy as np
from real2sim2real.artifacts import digest, write_json
from .guided_motion import select_interval, load_reference


def declared_hold(original_hold_s, reference_duration_s, requested_hold_s=None):
    """Extend the command duration explicitly, retaining native reference time."""
    hold = original_hold_s if requested_hold_s is None else requested_hold_s
    if (
        isinstance(hold, bool)
        or not np.isfinite([original_hold_s, reference_duration_s, hold]).all()
        or original_hold_s <= 0
        or reference_duration_s <= 0
        or not original_hold_s <= hold <= 25
        or hold < reference_duration_s + 1
    ):
        raise ValueError(
            "Explicit non-shortened hold must cover native motion plus one second, within 25 seconds"
        )
    return float(hold)


def match_window(old, new, *, allow_prepend=False):
    """Keep captured poses/timestamps intact; an earlier origin is explicit."""
    if type(allow_prepend) is not bool:
        raise ValueError("Reference prepend must be an explicit Boolean")
    indices = np.flatnonzero(new.frame_ids == old.frame_ids[0])
    start = int(indices[0]) if len(indices) == 1 else -1
    n = len(old.times)
    if (
        len(new.times) <= n
        or start < 0
        or start + n > len(new.times)
        or (start != 0 and not allow_prepend)
        or not np.array_equal(new.frame_ids[start : start + n], old.frame_ids)
        or not np.allclose(new.times[start : start + n], old.times, atol=1e-8, rtol=0)
        or not np.allclose(new.poses[start : start + n], old.poses, atol=1e-8, rtol=0)
    ):
        raise ValueError(
            "New window must preserve admitted capture poses and native timestamps"
        )
    return dict(
        prefix_preserved=start == 0,
        reference_origin_changed=start != 0,
        capture_origin_shift_s=float(old.times[0] - new.times[0]),
        original_capture_start_pose_reproduced=False,
    )


def reference_transforms(parent, settings):
    """Kinematic retargeting only: replace object increments, retain relative grasp."""
    old = load_reference(parent["configuration"]["reference_guidance"])
    new = load_reference(settings)
    n = len(old.times)
    if (
        len(new.times) <= n
        or not np.array_equal(new.frame_ids[:n], old.frame_ids)
        or not np.allclose(new.poses[:n], old.poses, atol=1e-8, rtol=0)
        or not np.allclose(new.times[:n], old.times, atol=1e-8, rtol=0)
    ):
        raise ValueError("New reference must preserve the admitted prefix")
    anchor = np.asarray(parent["reference_anchor_world_from_object"])
    changes, overrides = [], []
    for row in parent["records"]:
        change, override = np.eye(4), {}
        if row["phase"] == "hold":
            elapsed = row["reference_elapsed_s"]
            p, r = new.sample(elapsed)
            goal = anchor.copy()
            goal[:3, 3] += p
            goal[:3, :3] = r @ anchor[:3, :3]
            previous = np.eye(4)
            previous[:3, 3], previous[:3, :3] = (
                row["reference_position_m"],
                row["reference_rotation"],
            )
            change = goal @ np.linalg.inv(previous)
            pose = np.eye(4)
            pose[:3, 3], pose[:3, :3] = row["object_position_m"], row["object_rotation"]
            pose = change @ pose
            override = dict(
                reference_position_m=goal[:3, 3].tolist(),
                reference_rotation=goal[:3, :3].tolist(),
                object_position_m=pose[:3, 3].tolist(),
                object_rotation=pose[:3, :3].tolist(),
            )
        changes.append(change)
        overrides.append(override)
    if (
        max(r.get("reference_elapsed_s", 0) for r in parent["records"])
        < new.elapsed[-1] + 1
    ):
        raise ValueError(
            "Saved command clock does not cover extension and one second of hold"
        )
    return np.array(changes), overrides


def effective_parent(parent, audit):
    """Apply explicitly declared planning-reference overrides, never observed data."""
    if "reference_extension" not in audit["config"]:
        return parent
    parent = deepcopy(parent)
    overrides = audit["extended_reference_records"]
    if len(overrides) != len(parent["records"]):
        raise ValueError("Extended reference must cover every source sample")
    for row, override in zip(parent["records"], overrides):
        if set(override) - {
            "reference_position_m",
            "reference_rotation",
            "object_position_m",
            "object_rotation",
        }:
            raise ValueError(
                "Reference extension may not change phase, clock or fingers"
            )
        row.update(override)
    parent["configuration"]["reference_guidance"] = audit["config"][
        "reference_extension"
    ]
    return parent


def run(config):
    from .object_grasp_pilot import trial

    for path, sha in config["sources"].items():
        if digest(Path(path)) != sha:
            raise ValueError("Reference-extension source changed")
    parent_path = Path(config["parent_trial"])
    parent = json.loads(parent_path.read_text())
    if (
        not parent["success"]
        or digest(parent_path.with_name("states.npz")) != parent["states_sha256"]
    ):
        raise ValueError("Pinned accepted grasp and matching states required")
    old = load_reference(parent["configuration"]["reference_guidance"])
    with np.load(config["object_reference"], allow_pickle=False) as archive:
        ref = select_interval(archive, config["first_frame"], config["last_frame"])
    window = match_window(old, ref, allow_prepend=config.get("allow_prepend", False))
    out = Path(config["output"])
    out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        out / "reference.npz",
        timestamps_s=ref.times,
        world_from_object=ref.poses,
        frame_ids=ref.frame_ids,
    )
    settings = deepcopy(parent["configuration"])
    settings["reference_guidance"].update(
        path=str(out / "reference.npz"),
        sha256=digest(out / "reference.npz"),
        limits=config["motion_limits"],
    )
    load_reference(settings["reference_guidance"])
    settings["rollout"]["hold_s"] = declared_hold(
        parent["configuration"]["rollout"]["hold_s"],
        float(ref.elapsed[-1]),
        config.get("hold_s"),
    )
    write_json(
        out / "protocol.json",
        dict(
            config=config,
            **window,
            bounds=ref.bounds(),
            role="Cartesian fixture command-generation trial; requires a new arm audit and actuator-driven replay",
            physical_acceptance_criteria_changed=False,
            original_hold_s=parent["configuration"]["rollout"]["hold_s"],
            declared_hold_s=settings["rollout"]["hold_s"],
            reference_retimed=False,
            implementation_sha256=digest(Path(__file__)),
        ),
    )
    result = trial(settings, parent["candidate"], out / "source-trial")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--worker", action="store_true")
    a = p.parse_args()
    config = json.loads(a.config.read_text())
    if not a.worker:
        from real2sim2real.bounded_process import run as bounded

        return bounded(
            [
                sys.executable,
                "-m",
                __spec__.name,
                "--config",
                str(a.config.resolve()),
                "--worker",
            ],
            timeout=min(config["deadline_s"], 900),
        )
    run(config)


if __name__ == "__main__":
    main()
