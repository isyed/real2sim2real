"""Distinguish upstream source references from tested simulator adaptations."""

import json
from importlib.resources import files
from pathlib import Path

ADAPTATIONS = {
    "observe_environment": ["verifying-grasps"],
    "fit_grasp": ["grasping-direct-ik", "selecting-reachable-grasp"],
    "acquire_grasp": ["grasping-direct-ik", "grasping-with-planner"],
    "plan_transport": ["planning-held-object-motion", "registering-held-objects"],
    "execute_transport": ["executing-held-object-motion", "transporting-objects"],
    "verify_grasp": ["verifying-grasps"],
    "place_and_release": ["placing-held-feature", "transporting-objects"],
    "verify_placement": ["verifying-placement"],
}


def skill_catalog(available_tools):
    package = Path(str(files("real2sim2real.agentic_policy")))
    root = package / "_vendor/robot_skills"
    manifest = json.loads((root / "manifest.json").read_text())
    guide = package / "resources/skill-adaptations.md"
    if not guide.is_file():
        guide = package.parents[3] / "SKILLS.md"
    sources = []
    for name, row in sorted(manifest["skills"].items()):
        gap = row["gap"]
        required = set(gap.get("allowed_tools", []))
        required.update(gap.get("requires", {}).get("connector", []))
        sources.append(
            {
                "name": name,
                "description": row["description"],
                "license": row["license"],
                "definition": str(root / row["definition"]),
                "source_directory": str((root / row["definition"]).parent),
                "status": "source_reference_requires_adaptation",
                "reason": "Verbatim upstream imports and connector semantics are not registered here. Use an explicit adaptation or implement against tools.json; source availability is not execution validation.",
                "missing_declared_tools": sorted(required - set(available_tools)),
                "adaptations": [
                    "robot." + op for op, names in ADAPTATIONS.items() if name in names
                ],
            }
        )
    operations = [
        {
            "tool": "robot." + name,
            "upstream_references": names,
            "status": "executable_adaptation"
            if "robot." + name in available_tools
            else "backend_unavailable",
            "scope": "Project implementation for the continuous articulated-hand simulator; not an unchanged upstream skill, task success or control certification.",
        }
        for name, names in ADAPTATIONS.items()
    ]
    return {
        "schema_version": 1,
        "adaptation_guide": str(guide),
        "repository": manifest["repository"],
        "revision": manifest["revision"],
        "manifest": str(root / "manifest.json"),
        "license": str(root / "LICENSE"),
        "sources": sources,
        "adapted_operations": operations,
        "limits": "Read all bundled definitions/scripts. Optional perception models and external tool servers are not installed. Adaptations use privileged simulation observations and conditional geometry; no hardware connector is provided.",
    }
