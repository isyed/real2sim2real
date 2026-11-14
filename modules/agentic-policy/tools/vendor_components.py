import argparse
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser(
    description="Explicit maintainer-only upstream snapshot refresh"
)
parser.add_argument("--sources", type=Path, required=True)
args = parser.parse_args()
root = args.sources.resolve()
dest = Path(__file__).resolve().parents[1] / "src/real2sim2real/agentic_policy/_vendor"
dest.mkdir(exist_ok=True)
(dest / "__init__.py").write_text(
    '"""Pinned upstream components. See provenance.json and bundled license files."""\n'
)
manifest = {}
for name, repo, src, selected in [
    (
        "gap",
        "graph-as-policy",
        "gap",
        ["runtime", "skills", "__init__.py", "env_config.py"],
    ),
    ("gap_core", "graph-as-policy", "gap-core/src/gap_core", None),
    (
        "geometry",
        "open-robot-skills",
        "tools/geometry",
        ["tools.py", "_impl.py", "SKILL.md"],
    ),
    ("asl_prompts", "agentic-self-learning", "asl/prompts", None),
]:
    base = root / repo / src
    target = dest / name
    files = {}
    for p in base.rglob("*"):
        rel = p.relative_to(base)
        if (
            not p.is_file()
            or "__pycache__" in rel.parts
            or p.suffix not in {".py", ".md", ".json", ".yaml", ".yml"}
        ):
            continue
        if selected and rel.parts[0] not in selected:
            continue
        raw = p.read_text()
        new = raw
        for a, b in [
            (
                "gap_skills.tools.geometry",
                "real2sim2real.agentic_policy._vendor.geometry",
            ),
            ("gap_core", "real2sim2real.agentic_policy._vendor.gap_core"),
            ("gap", "real2sim2real.agentic_policy._vendor.gap"),
        ]:
            # Namespace relocation in imports and dynamic module names; no global aliases.
            new = re.sub(r"(?<![\w.])" + re.escape(a) + r"(?=\.|\b)", b, new)
        if name == "geometry" and str(rel) == "tools.py":
            # AttachmentResult must resolve when tools inspect return type hints.
            new = new.replace("from typing import TypedDict", "from typing import Any, TypedDict")
        q = target / rel
        q.parent.mkdir(parents=True, exist_ok=True)
        q.write_text(new)
        files[str(rel)] = {
            "source_sha256": hashlib.sha256(raw.encode()).hexdigest(),
            "bundled_sha256": hashlib.sha256(new.encode()).hexdigest(),
        }
    target.mkdir(exist_ok=True)
    if not (target / "__init__.py").exists():
        (target / "__init__.py").write_text("")
    for license in ["LICENSE", "NOTICE.md"]:
        if (root / repo / license).is_file():
            shutil.copyfile(root / repo / license, target / license)
    modifications = "Module namespace relocation; source text including docs follows namespace. ASL prompts retained as provenance, adapted campaign uses its own task-specific brief."
    if name == "geometry":
        modifications += " Import typing.Any for AttachmentResult annotation resolution."
    manifest[name] = {
        "repository": repo,
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root / repo
        )
        .decode()
        .strip(),
        "source_root": src,
        "files": files,
        "modifications": modifications,
    }
(dest / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")
print({k: len(v["files"]) for k, v in manifest.items()})
