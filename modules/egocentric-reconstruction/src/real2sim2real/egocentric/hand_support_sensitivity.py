"""Evaluate frozen meshes on several image-defined depth support margins.

No fitting or margin selection occurs here. All variants share exactly the same
reserved depth samples at each margin; the original support is also reported.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh

from real2sim2real.artifacts import digest, write_json
from .hand_depth_support import load_boundaries, support_region
from .hand_observation_fit import measured_surface
from .interaction_io import Reconstruction, fresh_directory
from .mesh_surface import MeshSurfaceQuery


def audit(config):
    source = Reconstruction(
        config["reconstruction"], config["frame_count"], start=config["frame_start"]
    )
    boundaries, receipt = load_boundaries(
        config["boundaries"],
        source.frame_ids,
        source.times,
        config["blocked_frame_ids"],
    )
    crops = dict(np.load(config["crops"]))
    raw = dict(np.load(config["raw"]))
    variants = {
        name: dict(np.load(spec["path"])) for name, spec in config["variants"].items()
    }
    for data in (crops, raw, *variants.values()):
        if not np.array_equal(data["frame_ids"], source.frame_ids) or not np.allclose(
            data["timestamps_s"], source.times, rtol=0, atol=1e-8
        ):
            raise ValueError("All sensitivity inputs must share the capture timeline")
    out = fresh_directory(config["output"])
    hashes = {
        p: digest(p)
        for p in [
            config["crops"],
            config["raw"],
            *[spec["path"] for spec in config["variants"].values()],
        ]
    }
    rows = []
    for frame in config["frame_ids"]:
        found = np.flatnonzero(source.frame_ids == frame)
        if len(found) != 1:
            raise ValueError("Sensitivity frame absent from capture")
        i = int(found[0])
        depth = source.depth(i)
        for h in range(2):
            mask = source.mask(i, config["mask_ids"][h])
            queries = {}
            for name, data in variants.items():
                vertices = data[
                    config["variants"][name].get("vertices_key", "camera_vertices")
                ][i, h]
                if np.isfinite(vertices).all():
                    queries[name] = MeshSurfaceQuery(
                        trimesh.Trimesh(vertices, raw["faces"][h], process=False)
                    )
            for margin in [None, *config["margins_px"]]:
                region = (
                    None
                    if margin is None
                    else support_region(depth.shape, boundaries, i, h, margin)
                )
                _, reserved = measured_surface(
                    depth,
                    mask,
                    crops["boxes_xyxy"][i, h],
                    source.k,
                    support_mask=region,
                )
                metrics = {}
                for name, query in queries.items():
                    distances = (
                        query.closest(reserved)[1] if len(reserved) else np.array([])
                    )
                    metrics[name] = dict(
                        median_m=float(np.median(distances))
                        if len(distances)
                        else None,
                        p90_m=float(np.quantile(distances, 0.9))
                        if len(distances)
                        else None,
                    )
                rows.append(
                    dict(
                        frame_id=int(frame),
                        hand=h,
                        margin_px=margin,
                        boundary_applied=margin is not None
                        and bool(boundaries["available"][i, h]),
                        reserved_points=len(reserved),
                        metrics=metrics,
                    )
                )
    report = dict(
        schema="real2sim2real.hand-support-sensitivity/1",
        boundary_receipt=receipt,
        source_hashes=hashes,
        margins_px=config["margins_px"],
        rows=rows,
        refitting_performed=False,
        limitation="Sensitivity of fixed hypotheses to an image support proxy; no anatomical or control certification and no margin selected from residuals.",
    )
    source.receipt(out, mask_ids=config["mask_ids"], config=config)
    report["reconstruction_receipt_sha256"] = digest(out / "inputs.json")
    write_json(out / "sensitivity.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    audit(json.loads(args.config.read_text()))


if __name__ == "__main__":
    main()
