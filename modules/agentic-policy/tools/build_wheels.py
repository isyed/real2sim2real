"""Build the four project wheels required for a standalone MuJoCo campaign."""

import argparse
import subprocess
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parents[3]
packages = [
    root,
    *(
        root / "modules" / name
        for name in ["agentic-policy", "controller-actor-training", "retargeting"]
    ),
]
subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "wheel",
        "--no-deps",
        "--wheel-dir",
        str(args.output.resolve()),
        *map(str, packages),
    ],
    check=True,
)
