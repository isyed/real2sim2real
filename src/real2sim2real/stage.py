"""Small common request reader; heavy libraries stay inside module execution."""

import argparse
import json
from pathlib import Path


def request_main(run):
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    run(request)


def option_path(request, name, *, required=True):
    value = request["options"].get(name)
    if value is None:
        if required:
            raise ValueError("Missing explicit module option: " + name)
        return None
    return (Path(request["config_base"]) / Path(value).expanduser()).resolve()
