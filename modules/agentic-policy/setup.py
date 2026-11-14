"""Build-time source reference; no neighboring repository is needed at runtime."""

import hashlib
import json
import shutil
import zipfile
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py
from setuptools.command.sdist import sdist

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def reference(destination):
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(HERE / "SKILLS.md", destination / "skill-adaptations.md")
    names = ["project-source.zip", "project-source.json"]
    if not (ROOT / "src/real2sim2real/cli.py").is_file():
        for name in names:
            shutil.copyfile(HERE / "_source_bundle" / name, destination / name)
        return
    extensions = {
        ".py",
        ".md",
        ".toml",
        ".json",
        ".yaml",
        ".yml",
        ".sh",
        ".cpp",
        ".cc",
        ".c",
        ".h",
        ".hpp",
        ".cu",
        ".cuh",
        ".cmake",
        ".proto",
        ".txt",
    }
    skipped = {
        ".git",
        "__pycache__",
        "build",
        "dist",
        ".venv",
        "node_modules",
        "weights",
        "checkpoints",
    }
    files = {}
    with zipfile.ZipFile(destination / names[0], "w", zipfile.ZIP_DEFLATED) as archive:
        for folder in ["src", "modules", "docs", "licenses", "scripts", "tools", "examples"]:
            for path in sorted((ROOT / folder).rglob("*")):
                if not path.is_file() or path.is_symlink() or path.is_relative_to(HERE):
                    continue
                rel = path.relative_to(ROOT)
                if set(rel.parts) & skipped or any(
                    part.endswith(".egg-info") for part in rel.parts
                ):
                    continue
                if (
                    path.suffix not in extensions
                    and "LICENSE" not in path.name
                    and "NOTICE" not in path.name
                ):
                    continue
                raw = path.read_bytes()
                entry = zipfile.ZipInfo(str(rel))
                entry.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(entry, raw)
                files[str(rel)] = hashlib.sha256(raw).hexdigest()
        for name in ["README.md", "LICENSE", "NOTICE"]:
            raw = (ROOT / name).read_bytes()
            entry = zipfile.ZipInfo(name)
            entry.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(entry, raw)
            files[name] = hashlib.sha256(raw).hexdigest()
    (destination / names[1]).write_text(
        json.dumps(
            {
                "files": files,
                "scope": "Source reference only; optional execution dependencies are separate. Agentic module source is installed directly.",
            },
            indent=2,
        )
        + "\n"
    )


class Build(build_py):
    def run(self):
        super().run()
        reference(Path(self.build_lib) / "real2sim2real/agentic_policy/resources")


class Source(sdist):
    def make_release_tree(self, base_dir, files):
        super().make_release_tree(base_dir, files)
        reference(Path(base_dir) / "_source_bundle")


setup(cmdclass={"build_py": Build, "sdist": Source})
