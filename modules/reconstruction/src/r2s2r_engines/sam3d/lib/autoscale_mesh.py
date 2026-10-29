# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Apply SAM3D's estimated object scale to its canonical output mesh."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import trimesh


def autoscale_mesh(
    mesh_path: str | Path,
    transform_path: str | Path,
    output_path: str | Path,
    report_path: str | Path | None = None,
) -> Path:
    mesh_path, transform_path, output_path = map(
        lambda value: Path(value).expanduser().resolve(),
        (mesh_path, transform_path, output_path),
    )
    report_path = (
        Path(report_path).expanduser().resolve()
        if report_path is not None
        else output_path.with_suffix(".scale.json")
    )
    if not mesh_path.is_file():
        raise FileNotFoundError(mesh_path)
    if not transform_path.is_file():
        raise FileNotFoundError(transform_path)
    if output_path.suffix.lower() != ".glb":
        raise ValueError(f"Output mesh must be GLB: {output_path}")

    transform = json.loads(transform_path.read_text())
    scale = np.asarray(transform.get("scale"), dtype=np.float64)
    if scale.shape != (3,) or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError(f"SAM3D transform must contain three finite positive scale values: {scale}")

    loaded = trimesh.load(mesh_path, force="scene", process=False)
    if isinstance(loaded, trimesh.Trimesh):
        scene = trimesh.Scene(loaded)
    elif isinstance(loaded, trimesh.Scene):
        scene = loaded
    else:
        raise TypeError(f"SAM3D output did not load as a mesh or scene: {mesh_path}")
    meshes = [geometry for geometry in scene.geometry.values() if isinstance(geometry, trimesh.Trimesh)]
    if not meshes or any(len(mesh.vertices) == 0 or len(mesh.faces) == 0 for mesh in meshes):
        raise ValueError(f"SAM3D output contains no usable triangle geometry: {mesh_path}")
    bounds_before = np.asarray(scene.bounds, dtype=np.float64)
    if bounds_before.shape != (2, 3) or not np.isfinite(bounds_before).all():
        raise ValueError(f"SAM3D output has invalid bounds: {bounds_before}")

    scale_transform = np.eye(4, dtype=np.float64)
    scale_transform[:3, :3] = np.diag(scale)
    scene.apply_transform(scale_transform)

    payload = scene.export(file_type="glb")
    if not isinstance(payload, bytes) or not payload:
        raise ValueError(f"Scaled GLB export is empty: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    check = trimesh.load(temporary, file_type="glb", force="scene", process=False)
    bounds_after = np.asarray(check.bounds, dtype=np.float64)
    expected_bounds = bounds_before * scale[None, :]
    tolerance = max(float(np.max(np.abs(expected_bounds))) * 1e-5, 1e-7)
    if (
        not isinstance(check, trimesh.Scene)
        or bounds_after.shape != (2, 3)
        or not np.isfinite(bounds_after).all()
        or np.max(np.abs(bounds_after - expected_bounds)) > tolerance
    ):
        temporary.unlink(missing_ok=True)
        raise ValueError(
            f"Scaled GLB round-trip has unexpected bounds: expected={expected_bounds}, actual={bounds_after}"
        )
    os.replace(temporary, output_path)

    report = {
        "schema": "r2s2r_engines.sam3d_mesh_autoscale.v1",
        "source_mesh": str(mesh_path),
        "source_transform": str(transform_path),
        "output_mesh": str(output_path),
        "scale_xyz": scale.tolist(),
        "source_bounds": bounds_before.tolist(),
        "scaled_bounds": bounds_after.tolist(),
        "scale_source": "SAM3D image-to-mesh transform estimate; not a calibrated scan",
    }
    report_tmp = report_path.with_name(f".{report_path.name}.{os.getpid()}.tmp")
    report_tmp.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(report_tmp, report_path)
    print(json.dumps(report, indent=2, sort_keys=True))
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh_path", required=True, help="Canonical GLB produced by SAM3D")
    parser.add_argument("--transform_path", required=True, help="SAM3D transform JSON")
    parser.add_argument("--output_path", required=True, help="Metric-scale candidate GLB for CARI4D")
    parser.add_argument("--report_path", help="Autoscaling report path; defaults beside output_path")
    args = parser.parse_args()
    autoscale_mesh(args.mesh_path, args.transform_path, args.output_path, args.report_path)


if __name__ == "__main__":
    main()
