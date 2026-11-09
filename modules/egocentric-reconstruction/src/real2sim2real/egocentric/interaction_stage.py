"""Optional orchestrator adapter; partial hands never become full-body motion."""

from real2sim2real.stage import request_main
from .interaction import execute


def run(request):
    config = dict(
        request["options"], reconstruction=request["input"], output=request["output"]
    )
    # The caller authorizes this bounded run; hands-check loads on CPU first.
    for phase in (
        "support",
        "hands2d",
        "hands-check",
        "hands",
        "align-hands",
        "contacts",
        "score",
        "review",
        "publish",
    ):
        execute(config, phase)


if __name__ == "__main__":
    request_main(run)
