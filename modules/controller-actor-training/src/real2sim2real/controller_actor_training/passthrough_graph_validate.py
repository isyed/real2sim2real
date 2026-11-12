"""Bounded CPU verification of a pinned GaP pass-through against frozen physics."""

import argparse
import importlib.metadata
import json
import os
import sys
from copy import deepcopy
from pathlib import Path

from real2sim2real.artifacts import digest, write_json

from .arm_residual_env import ArmResidualEnv, frozen_source
from .arm_residual_validate import compare
from .continuous_residual import validate_baseline
from .passthrough_graph import ZeroPhaseSession, execute, phase_order, validate_workflow


def environment_config(config):
    required = {
        "baseline_result",
        "baseline_sha256",
        "output",
        "deadline_s",
        "policy_period_s",
        "workflow",
        "workflow_sha256",
    }
    if set(config) != required:
        raise ValueError("Pinned baseline/workflow, output and bounded clocks required")
    if isinstance(config["deadline_s"], bool) or not 0 < config["deadline_s"] <= 900:
        raise ValueError("A finite wall deadline up to 900 seconds is required")
    return {
        "baseline_result": config["baseline_result"],
        "baseline_sha256": config["baseline_sha256"],
        "output": str(Path(config["output"]) / "replay"),
        "deadline_s": config["deadline_s"],
        "residual": {
            "policy_period_s": config["policy_period_s"],
            "max_offset_n": 1.0,
            "max_rate_n_s": 5.0,
            "active_phases": ["close", "lift", "hold", "return"],
        },
    }


def source_files(path):
    """Verify and collect the pinned evidence chain, including audited assets."""
    path = Path(path).resolve()
    report = json.loads(path.read_text())
    pending = {
        str(path): digest(path),
        str(path.with_name("scene.xml")): report["scene_sha256"],
        str(path.with_name("states.npz")): report["states_sha256"],
    }
    collected = {}
    while pending:
        name, sha = pending.popitem()
        if name in collected:
            if collected[name] != sha:
                raise ValueError("Conflicting pinned evidence hashes")
            continue
        source = Path(name)
        if digest(source) != sha:
            raise ValueError(f"Pinned input changed: {source.name}")
        collected[name] = sha
        if source.suffix == ".json":
            document = json.loads(source.read_text())
            if not isinstance(document, dict):
                continue
            for mapping in (
                document.get("sources", {}),
                document.get("config", {}).get("sources", {}),
            ):
                if not isinstance(mapping, dict):
                    continue
                for filename, expected in mapping.items():
                    if isinstance(expected, str) and len(expected) == 64:
                        filename = str(Path(filename).resolve())
                        if filename in pending and pending[filename] != expected:
                            raise ValueError("Conflicting pending evidence hashes")
                        pending[filename] = expected
    return dict(sorted(collected.items()))


def package_receipt(name):
    distribution = importlib.metadata.distribution(name)
    direct = distribution.read_text("direct_url.json")
    return {
        "version": distribution.version,
        "source": json.loads(direct) if direct else None,
    }


def run(config):
    config = deepcopy(config)
    env_config = environment_config(config)
    baseline = frozen_source(env_config)
    validate_baseline(baseline)
    phases = phase_order(baseline["records"])
    graph_path = Path(config["workflow"])
    if digest(graph_path) != config["workflow_sha256"]:
        raise ValueError("Pinned graph changed")
    graph = json.loads(graph_path.read_text())
    validate_workflow(graph, phases)
    pinned = source_files(config["baseline_result"])
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=False)
    graph_copy = output / "workflow" / "workflow.json"
    graph_copy.parent.mkdir()
    graph_copy.write_bytes(graph_path.read_bytes())
    implementation = output / "implementation"
    implementation.mkdir()
    hashes = {}
    for name in ("passthrough_graph.py", "passthrough_graph_validate.py"):
        target = implementation / name
        target.write_bytes(Path(__file__).with_name(name).read_bytes())
        hashes[name] = digest(target)
    import mujoco
    import numpy
    import scipy

    manifest = {
        "schema": "real2sim2real.frozen-graph-inputs/1",
        "config": config,
        "source_files": pinned,
        "workflow_sha256": digest(graph_copy),
        "implementation_sha256": hashes,
        "runtime": {
            "python": sys.version,
            "mujoco": mujoco.__version__,
            "numpy": numpy.__version__,
            "scipy": scipy.__version__,
            "gap": package_receipt("graph-as-policy"),
            "gap_core": package_receipt("graph-as-policy-core"),
        },
        "controller_config": baseline["config"],
        "continuous_protocol": baseline["continuous_protocol"],
        "phases": list(phases),
        "physics_timestep_s": baseline["physics_timestep_s"],
        "controller_period_s": baseline["controller_period_s"],
        "policy_period_s": config["policy_period_s"],
        "policy_generation": False,
        "skill_autodiscovery": False,
        "training_run": False,
    }
    write_json(output / "manifest.json", manifest)
    session = ZeroPhaseSession(lambda: ArmResidualEnv(env_config), phases)
    try:
        graph_status = execute(graph_copy, session, output / "graph-trace")
        if digest(graph_copy) != config["workflow_sha256"]:
            raise ValueError("Executed graph changed")
        if pinned != source_files(config["baseline_result"]):
            raise ValueError("Frozen sources changed during execution")
        verification = compare(
            config["baseline_result"], session.env.directory / "result.json"
        )
        boundaries = session.receipt()
        verification.update(
            schema="real2sim2real.passthrough-graph-verification/1",
            graph_exit_status=graph_status,
            graph=boundaries,
            workflow_sha256=config["workflow_sha256"],
            manifest_sha256=digest(output / "manifest.json"),
            pinned_sources_unchanged=True,
            policy_generation=False,
            graph_completion_is_task_success=False,
        )
        verification["pass_through_reproduction_passed"] = (
            verification["zero_reproduction_passed"]
            and boundaries["complete"]
            and boundaries["boundary_clock_preserved"]
            and boundaries["reset_count"] == 1
        )
        write_json(output / "verification.json", verification)
        print(
            json.dumps(
                {
                    k: verification[k]
                    for k in (
                        "pass_through_reproduction_passed",
                        "baseline_success",
                        "candidate_success",
                        "rejected_for",
                    )
                }
            ),
            flush=True,
        )
        if not verification["pass_through_reproduction_passed"]:
            raise RuntimeError("Graph changed the frozen controller or timeline")
        return verification
    except BaseException as error:
        write_json(
            output / "interrupted.json",
            {
                "reason": type(error).__name__,
                "detail": str(error),
                "pass_through_reproduction_passed": False,
                "graph": session.receipt(),
                "training_eligible": False,
            },
        )
        raise
    finally:
        session.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    environment_config(config)
    if args.worker:
        run(config)
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
