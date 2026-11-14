import hashlib
import json
from pathlib import Path

from real2sim2real.agentic_policy.skill_catalog import ADAPTATIONS, skill_catalog


def test_all_upstream_sources_are_pinned_readable_and_not_advertised_as_runnable():
    catalog = skill_catalog({"robot." + name for name in ADAPTATIONS})
    manifest = json.loads(Path(catalog["manifest"]).read_text())
    root = Path(catalog["manifest"]).parent
    assert catalog["revision"] == "752dfbd96968d4d08379f8111765efbf6457800b"
    assert len(catalog["sources"]) == 33
    assert len(manifest["files"]) == 145
    for name, expected in manifest["files"].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected
    for skill in catalog["sources"]:
        assert Path(skill["definition"]).is_file()
        assert skill["status"] == "source_reference_requires_adaptation"
        assert skill["license"]
    assert all(
        r["status"] == "executable_adaptation" for r in catalog["adapted_operations"]
    )
    assert all(
        r["status"] == "backend_unavailable"
        for r in skill_catalog({})["adapted_operations"]
    )


def test_gripper_and_motion_connectors_are_not_silently_aliased():
    catalog = skill_catalog({"robot.observe", "robot.release", "robot.plan_wrist"})
    grasp = next(s for s in catalog["sources"] if s["name"] == "grasping-direct-ik")
    assert "robot.close_gripper" in grasp["missing_declared_tools"]
    assert "robot.go_to_pose" in grasp["missing_declared_tools"]
