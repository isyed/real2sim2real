"""Copy saved control evidence into typed artifacts without changing its source."""

import json
from pathlib import Path
import shutil

from .artifacts import digest, publish


def adopt(source, output, kind):
    source, output = Path(source).resolve(), Path(output).resolve()
    if kind == "grasp-trial":
        value = json.loads((source / "result.json").read_text())
        if value.get("schema") != "real2sim2real.object-grasp-trial/2":
            raise ValueError("Expected a saved object-grasp trial")
        if digest(source / "states.npz") != value["states_sha256"]:
            raise ValueError("Trial state archive changed")
        files = {"result": "result.json", "states": "states.npz"}
        accepted = value["success"] is True
    elif kind == "arm-path-audit":
        value = json.loads((source / "audit.json").read_text())
        if value.get("schema") != "real2sim2real.arm-path-audit/1":
            raise ValueError("Expected a saved arm-path audit")
        receipt = json.loads((source / "assembly.json").read_text())
        if digest(source / "audit-scene.xml") != receipt["scene_sha256"]:
            raise ValueError("Assembled arm scene changed")
        for path, sha in value["sources"].items():
            if digest(path) != sha:
                raise ValueError("Arm audit dependency changed: " + path)
        files = {
            "audit": "audit.json",
            "scene": "audit-scene.xml",
            "assembly": "assembly.json",
        }
        accepted = any(p["screen_pass"] for p in value["placements"])
    else:
        raise ValueError("Unsupported control artifact kind")
    output.mkdir(parents=True, exist_ok=False)
    for name in files.values():
        shutil.copyfile(source / name, output / name)
    return publish(
        output,
        kind,
        {r: output / n for r, n in files.items()},
        metadata={
            "source_directory": str(source),
            "source_stage_accepted": accepted,
            "external_dependencies_required": True,
            "training_ready_established": False,
        },
    )
