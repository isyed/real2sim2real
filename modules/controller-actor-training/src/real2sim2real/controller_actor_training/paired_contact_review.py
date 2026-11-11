"""Review saved contact hypotheses against existing calibrated RGB/depth.

This reviews contact centres, not complete pad patches or independent ground
truth. Missing visibility is distinct from geometric/depth disagreement.
"""

import argparse
from collections import Counter
import json
import inspect
import shutil
import os
from pathlib import Path
import sys

import numpy as np

from real2sim2real.artifacts import digest, write_json
from .paired_acquisition_audit import pinned_json


def review(config):
    import trimesh
    from PIL import Image, ImageDraw
    from real2sim2real.egocentric.interaction_io import Reconstruction
    from real2sim2real.egocentric.camera import transform
    from real2sim2real.egocentric.contact_observations import (
        observe_contacts,
        summarize_contact,
    )
    from .object_grasp_pilot import load_scene, load_plan
    from real2sim2real.egocentric.contact_observations import visible_surface

    audit = pinned_json(config["acquisition_audit"])
    plan = pinned_json(audit["config"]["plan"])
    parent = pinned_json(plan["config"]["source_trial"])
    support = plan["candidates"][audit["config"]["candidate_index"]]
    primary = parent["candidate"]
    settings = parent["configuration"]
    _, _, mesh = load_scene(settings)
    settled = np.asarray(load_plan(settings)["settling"]["settled_world_from_object"])
    task = json.loads(Path(settings["task"]).read_text())
    source = Reconstruction(config["reconstruction"])
    if task["object"]["sha256"] != digest(source.path("mesh/metric.glb")):
        raise ValueError("Contact geometry and reconstruction differ")
    for name in ("geometry.npz", "timeline.json"):
        if digest(source.path(name)) not in task["sources"].values():
            raise ValueError("Contact scene and observations have different provenance")
    frames = config["frames"]
    if (
        not 1 <= len(frames) <= 40
        or len(set(frames)) != len(frames)
        or any(type(i) is not int or not 0 <= i < source.count for i in frames)
    ):
        raise ValueError("Require up to forty distinct capture frame indices")
    points = transform(
        np.vstack([primary["targets_m"], support["targets_m"]]), np.linalg.inv(settled)
    )
    _, _, faces = trimesh.proximity.closest_point(mesh, points[:5])
    normals = np.vstack(
        [
            mesh.face_normals[faces],
            np.asarray(support["outward_normals"]) @ settled[:3, :3],
        ]
    )
    labels = [
        h + "_" + f
        for h in ("primary", "support")
        for f in ("thumb", "index", "middle", "ring", "pinky")
    ]
    if len(points) != len(labels):
        raise ValueError("Require five anatomical pad targets per hand")
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=False)
    hashes = {
        str(source.path(n)): digest(source.path(n))
        for n in ("artifact.json", "geometry.npz", "timeline.json", "mesh/metric.glb")
    }
    helpers = [
        Path(__file__),
        Path(inspect.getfile(pinned_json)),
        Path(inspect.getfile(observe_contacts)),
        Path(inspect.getfile(Reconstruction)),
        Path(inspect.getfile(load_scene)),
    ]
    snapshot = output / "implementation"
    snapshot.mkdir()
    for path in helpers:
        hashes[str(path)] = digest(path)
        shutil.copyfile(path, snapshot / path.name)
    rows = [[] for _ in points]
    for i in frames:
        camera_from_object = (
            np.linalg.inv(source.world_from_camera[i]) @ source.world_from_object[i]
        )
        depth, mask = source.depth(i), source.mask(i, 1)
        occluders = source.mask(i, 2) | source.mask(i, 3)
        observations = observe_contacts(
            points, normals, camera_from_object, source.k, depth, mask, occluders
        )
        visible = visible_surface(
            mesh, points, normals, np.linalg.inv(camera_from_object), 0.002
        )
        image = Image.fromarray(source.rgb(i))
        draw = ImageDraw.Draw(image)
        for j, row in enumerate(observations):
            if row["observed"] and not visible[j]:
                row.update(
                    observed=False, reason="mesh_self_occlusion", residual_m=None
                )
            row.update(
                frame_index=i,
                frame_id=int(source.frame_ids[i]),
                timestamp_s=float(source.times[i]),
            )
            rows[j].append(row)
            if row["pixel"] is not None:
                x, y = row["pixel"]
                if 0 <= x < image.width and 0 <= y < image.height:
                    color = "#00ffff" if row["observed"] else "#ff9900"
                    draw.ellipse((x - 3, y - 3, x + 3, y + 3), outline=color, width=2)
                    draw.text((x + 4, y), labels[j], fill=color)
        image.save(output / f"frame-{i:03d}.jpg")
        for n in (
            f"rgb/{i:06d}.png",
            f"depth/{i:06d}.npy",
            *(f"masks/{m}/{i:06d}.png" for m in (1, 2, 3)),
        ):
            hashes[str(source.path(n))] = digest(source.path(n))
    contacts = [
        dict(
            label=label,
            point_object_m=point.tolist(),
            summary=summarize_contact(observations),
            reasons=dict(Counter(r["reason"] for r in observations)),
            observations=observations,
        )
        for label, point, observations in zip(labels, points, rows)
    ]
    result = dict(
        schema="paired-contact-review/1",
        config=config,
        contacts=contacts,
        source_hashes=hashes,
        evidence_scope="Existing reconstruction RGB/masks/depth; contact centres only; no fresh raw-stereo triangulation or independent ground truth",
        finite_patch_observation_validated=False,
        independent_observation_validated=False,
        control_admitted=False,
        baseline_replaced=False,
    )
    pinned_json(config["acquisition_audit"])
    pinned_json(audit["config"]["plan"])
    pinned_json(plan["config"]["source_trial"])
    if any(digest(Path(p)) != sha for p, sha in hashes.items()):
        raise ValueError("Review input changed during execution")
    write_json(output / "review.json", result)
    table = "".join(
        f"<tr><td>{r['label']}</td><td>{r['summary']['evidence']}</td><td>{r['summary']['observed_frames']}</td></tr>"
        for r in contacts
    )
    images = "".join(
        f'<figure><figcaption>Capture frame index {i}</figcaption><img style="max-width:100%" src="frame-{i:03d}.jpg"></figure>'
        for i in frames
    )
    (output / "review.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Paired contact evidence</title><h1>Paired contact evidence</h1><p>'
        + result["evidence_scope"]
        + ". Cyan: visible depth sample. Orange: unavailable/occluded. A sample is not a validation pass.</p><table><tr><th>Pad</th><th>Depth evidence</th><th>Observed frames</th></tr>"
        + table
        + "</table>"
        + images
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if not 0 < config["deadline_s"] <= 900:
        raise ValueError("Require a finite review deadline up to 900 seconds")
    if args.worker:
        review(config)
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
            timeout=config["deadline_s"],
            env={
                **os.environ,
                "CUDA_VISIBLE_DEVICES": "",
                "OPENBLAS_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
            },
        )


if __name__ == "__main__":
    main()
