"""Check first-party source syntax and accidental private-workspace coupling."""

from pathlib import Path
import ast
import re

root = Path(__file__).resolve().parents[1]
bad = []
count = 0
for directory in [root / "src", root / "modules"]:
    for path in directory.rglob("*.py"):
        if any(
            p in {"build", "__pycache__"} or p.endswith(".egg-info") for p in path.parts
        ):
            continue
        ast.parse(path.read_text(), filename=str(path))
        count += 1
        if re.search(
            r"/home/(?!\.\.\.)|gs://|(?:from|import) (?:grounding|robotic_grounding)\b",
            path.read_text(),
        ):
            bad.append(str(path.relative_to(root)))
if bad:
    raise SystemExit("Nonportable source paths/imports: " + ", ".join(bad))
print(f"{count} Python files parse; no private-workspace runtime imports found")
