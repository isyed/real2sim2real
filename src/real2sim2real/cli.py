"""Command line selection of independent modules or a configured pipeline."""

import argparse
import json
from .pipeline import stage_registry, read_config, plan, execute
from .artifacts import load


def main(argv=None):
    parser = argparse.ArgumentParser(prog="r2s2r")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("modules", help="List selectable modules")
    for name in ["plan", "run"]:
        p = sub.add_parser(name)
        p.add_argument("--config", required=True)
        p.add_argument("--stages", help="Comma-separated modules, in execution order")
        if name == "run":
            p.add_argument("--resume", action="store_true")
    p = sub.add_parser("inspect")
    p.add_argument("artifact")
    args = parser.parse_args(argv)
    try:
        if args.command == "modules":
            result = list(stage_registry())
        elif args.command == "inspect":
            result = load(args.artifact)
        else:
            selected = args.stages.split(",") if args.stages else None
            if args.command == "plan":
                config, base = read_config(args.config)
                result = plan(config, base, selected=selected)
            else:
                result = {
                    "output": str(
                        execute(args.config, selected=selected, resume=args.resume)
                    )
                }
        print(json.dumps(result, indent=2))
        return 0
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
