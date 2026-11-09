"""Orchestrator adapter; object-only output cannot masquerade as full-body motion."""

from real2sim2real.artifacts import role_path
from real2sim2real.stage import request_main
from .cli import execute


def run(request):
    config = dict(
        request["options"],
        capture=str(role_path(request["input"], "capture")),
        output=request["output"],
    )
    for phase in (
        "calibrate",
        "prepare",
        "segment",
        "camera",
        "mesh",
        "track",
        "publish",
    ):
        execute(config, phase)


if __name__ == "__main__":
    request_main(run)
