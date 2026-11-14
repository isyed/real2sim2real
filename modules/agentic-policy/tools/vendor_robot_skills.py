"""Snapshot committed upstream skills as reference resources, never runtime imports."""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import yaml


def snapshot(source, revision, destination):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(source), *args])

    commit = git("rev-parse", revision + "^{commit}").decode().strip()
    if revision != commit:
        raise ValueError("Supply a full immutable commit SHA")
    # Read Git objects, not working-tree files. Local edits cannot enter the bundle.
    paths = git("ls-tree", "-r", "--name-only", commit, "skills").decode().splitlines()
    notices = git("ls-tree", "--name-only", commit).decode().splitlines()
    paths += [p for p in notices if p.startswith(("LICENSE", "NOTICE"))]
    if "LICENSE" not in paths:
        raise ValueError("Upstream LICENSE is required")
    files, skills, content = {}, {}, {}
    for name in sorted(paths):
        raw = git("show", commit + ":" + name)
        content[name] = raw
        files[name] = hashlib.sha256(raw).hexdigest()
        if name.endswith("/SKILL.md"):
            front = yaml.safe_load(raw.decode().split("---", 2)[1])
            skill = Path(name).parent.name
            if front["name"] != skill:
                raise ValueError("Skill directory/name mismatch: " + name)
            skills[skill] = {
                "description": front["description"],
                "license": front.get("license", "see upstream LICENSE"),
                "definition": name,
                "gap": front.get("gap", {}),
            }
    manifest = {
        "repository": "https://github.com/graph-robots/open-robot-skills",
        "revision": commit,
        "modifications": "None. Verbatim committed skill definitions, scripts and references; not registered as executable upstream skills.",
        "files": files,
        "skills": skills,
    }
    if destination.exists():
        shutil.rmtree(destination)
    for name, raw in content.items():
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    dest = (
        Path(__file__).resolve().parents[1]
        / "src/real2sim2real/agentic_policy/_vendor/robot_skills"
    )
    result = snapshot(args.source, args.revision, dest)
    print(
        json.dumps(
            {
                "revision": result["revision"],
                "skills": len(result["skills"]),
                "files": len(result["files"]),
            }
        )
    )
