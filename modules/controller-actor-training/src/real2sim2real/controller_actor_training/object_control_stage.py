"""Pipeline adapters for admitted grasp paths and bounded arm-control trials."""

import json
import math
from pathlib import Path

from real2sim2real.artifacts import digest, load, publish, role_path, write_json
from real2sim2real.stage import option_path, request_main


def bind_config(request):
    """Bind immutable upstream artifacts instead of trusting a stale template path."""
    path = option_path(request, "config")
    if digest(path) != request["options"].get("config_sha256"):
        raise ValueError("Pin the stage configuration with config_sha256")
    config = json.loads(path.read_text())
    budget = config.get("deadline_s", 900)
    timeout = request.get("timeout_seconds", 900)
    if not math.isfinite(budget) or budget <= 0 or not 0 < timeout <= min(budget, 900):
        raise ValueError(
            "Stage timeout must respect the module deadline and 900-second pilot bound"
        )
    inp, out = Path(request["input"]), Path(request["output"])
    stage = request["stage"]
    if stage == "arm-path-audit":
        load(inp, kinds={"grasp-trial"})
        trial = role_path(inp, "result")
        parent = json.loads(trial.read_text())
        if (
            parent.get("success") is not True
            or digest(role_path(inp, "states")) != parent["states_sha256"]
        ):
            raise ValueError(
                "Arm integration requires an accepted grasp trial and matching states"
            )
        if role_path(inp, "states") != trial.with_name("states.npz"):
            raise ValueError("Grasp states must accompany their result")
        config["sources"].pop(config.get("trial"), None)
        config["trial"] = str(trial)
        config["sources"][str(trial)] = digest(trial)
    elif stage == "arm-replay":
        load(inp, kinds={"arm-path-audit"})
        audit = role_path(inp, "audit")
        value = json.loads(audit.read_text())
        selected = [
            p
            for p in value["placements"]
            if p["placement"]["name"] == config["placement"]
        ]
        if (
            len(selected) != 1
            or selected[0]["screen_pass"] is not True
            or value.get("retained_fixture_hands")
        ):
            raise ValueError(
                "Arm replay requires the selected complete arm path to pass"
            )
        if role_path(inp, "scene") != audit.with_name("audit-scene.xml") or role_path(
            inp, "assembly"
        ) != audit.with_name("assembly.json"):
            raise ValueError("Audit scene and assembly must accompany the audit")
        config.update(audit=str(audit), audit_sha256=digest(audit))
    else:
        raise ValueError("Unknown object-control stage")
    config["output"] = str(out / "run")
    return config


def run(request):
    config = bind_config(request)
    out = Path(request["output"])
    write_json(out / "bound-config.json", config)
    if request["stage"] == "arm-path-audit":
        from .arm_path_audit import run as audit

        audit(config)
        files = {
            "audit": "audit.json",
            "scene": "audit-scene.xml",
            "assembly": "assembly.json",
        }
        result = json.loads((out / "run/audit.json").read_text())
        accepted = any(p["screen_pass"] for p in result["placements"])
    else:
        from .arm_replay_pilot import run as replay

        replay(config)
        files = {"result": "result.json", "states": "states.npz", "scene": "scene.xml"}
        result = json.loads((out / "run/result.json").read_text())
        accepted = result["success"]
    publish(
        out,
        request["output_kind"],
        {k: out / "run" / v for k, v in files.items()},
        metadata={
            "stage_accepted": accepted,
            "training_ready_established": False,
            "human_fingers_are_required_targets": False,
            "hardware_validated": False,
        },
    )


if __name__ == "__main__":
    request_main(run)
