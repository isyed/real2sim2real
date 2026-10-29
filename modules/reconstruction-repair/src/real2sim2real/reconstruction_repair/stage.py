"""Pipeline adapter for metric preflight, gated object repair and optional human repair."""

from pathlib import Path
import json
import shutil
import numpy as np

from real2sim2real.artifacts import load, role_path, publish, digest, write_json
from real2sim2real.stage import option_path, request_main
from .bundle import BundleData, bundle_create, arrays_read, file_sha256
from .contracts import COORDINATES, RepairConfig
from .runner import repair_run


def run(request):
    source = Path(request["input"])
    out = Path(request["output"])
    options = request["options"]
    artifact = load(source, kinds={"reconstruction"})
    g = arrays_read(role_path(source, "geometry"))
    n = len(g["frame_ids"])
    human = None
    if "human" in artifact["files"]:
        packed = arrays_read(role_path(source, "human"))
        human = dict(
            pose=packed["mhr_model_params"][:, :136],
            scales=packed["mhr_model_params"][:, 136:],
            shape=packed["shape_params"],
            expression=packed.get("face_params", np.zeros((n, 72))),
        )
    observations_path = option_path(request, "observations", required=False)
    if observations_path:
        observations = arrays_read(observations_path)
    else:
        import h5py

        with (
            h5py.File(role_path(source, "object_masks")) as obj,
            h5py.File(role_path(source, "human_masks")) as body,
        ):
            observations = dict(
                object_mask=np.asarray(obj["frames"]) != 0,
                occluder_mask=np.asarray(body["frames"]) != 0,
            )
        if len(observations["object_mask"]) != n:
            raise ValueError("Observation timeline differs")
        observations.update(
            reliable=(observations["object_mask"] & ~observations["occluder_mask"])
            .reshape(n, -1)
            .sum(1)
            >= 100,
            locked=np.zeros(n, bool),
        )
    metadata = dict(
        sequence_id=artifact["metadata"]["sequence_id"],
        world_frame_id=artifact["metadata"]["world_frame_id"],
        coordinates=COORDINATES,
        evidence=options.get("evidence", {}),
    )
    if human is not None:
        metadata["human"] = {"encoding": "native_mhr_204"}
    if options.get("repair", {}).get("human_repair", False):
        if human is None:
            raise ValueError("Human repair requires native human parameters")
        # Independent exported parity is evidence, not a self-generated pass.
        parity = arrays_read(option_path(request, "human_parity"))
        if set(parity) != {"parity_frame_ids", "parity_vertices_world"}:
            raise ValueError(
                "human_parity must contain independently exported frame IDs and world vertices"
            )
        human.update(parity)
        metadata["human"] = json.loads(
            option_path(request, "human_metadata").read_text()
        )
    bundle = bundle_create(
        out / "input_bundle", BundleData(metadata, g, observations, human)
    )
    metric = option_path(request, "metric_evidence")
    floor_source = options.get("floor_roi_source")
    if not floor_source:
        raise ValueError(
            "Metric calibration requires an explicit reviewed floor ROI source"
        )
    proof = dict(
        schema="r2s2r_engines.metric-calibration-evidence/1",
        source=options.get("metric_evidence_source", "Predicted metric camera-Z depth"),
        floor_roi_source=floor_source,
        clean_trajectory_used=False,
        input_manifest_sha256=file_sha256(bundle.root / "manifest.json"),
        evidence_sha256=digest(metric),
    )
    write_json(out / "metric_provenance.json", proof)
    config = RepairConfig.from_dict(options.get("repair", {}))
    decoder = None
    if options.get("mhr_decoder"):
        from .human_decoder import TorchMHRDecoder

        decoder = TorchMHRDecoder(option_path(request, "mhr_decoder"))
    result = repair_run(
        bundle.root,
        out / "repair",
        config,
        human_decoder=decoder,
        metric_evidence=metric,
        metric_provenance=out / "metric_provenance.json",
    )
    # Use only the accepted bundle selected by the repair runner; rejected
    # proposals remain in diagnostic stage records.
    selected = result.output_bundle
    from .bundle import bundle_read

    selected = bundle_read(selected)
    np.savez_compressed(out / "geometry.npz", **selected.geometry)
    files = {
        "geometry": out / "geometry.npz",
        "diagnostics": out / "repair/report.json",
    }
    if human is not None:
        a = arrays_read(selected.root / selected.manifest["assets"]["human"]["path"])
        np.savez_compressed(
            out / "human.npz",
            mhr_model_params=np.concatenate(
                (a["pose"], np.broadcast_to(a["scales"], (n, 68))), axis=-1
            ),
            shape_params=a["shape"],
            face_params=a.get("expression", np.zeros((n, 72))),
        )
        files["human"] = out / "human.npz"
    for role in ["timeline", "depth", "object_masks", "human_masks"]:
        if role in artifact["files"]:
            old = role_path(source, role)
            target = out / (role + old.suffix)
            shutil.copy2(old, target)
            files[role] = target
    # Recreate mesh from the accepted vertices. Retargeting uses the explicit
    # geometry bundle, so texture loss here does not alter geometry/pose.
    import trimesh

    mesh = trimesh.Trimesh(
        selected.geometry["vertices"] * float(selected.geometry["object_scale"]),
        selected.geometry["faces"],
        process=False,
    )
    mesh.export(out / "mesh.glb")
    files["mesh"] = out / "mesh.glb"
    publish(
        out,
        "reconstruction",
        files,
        metadata=dict(
            artifact["metadata"],
            repair_status=result.report.get("status", "complete"),
            parent_artifact_sha256=digest(source / "artifact.json"),
        ),
    )


if __name__ == "__main__":
    request_main(run)
