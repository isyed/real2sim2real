"""Command line selection of independent modules or a configured pipeline."""

import argparse
import json
import sys
from .pipeline import stage_registry, read_config, plan, execute
from .artifacts import load
from .pipeline_profiles import PROFILES


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "agentic":
        try:
            from real2sim2real.agentic_policy.cli import main as agentic_main
        except ImportError as error:
            raise SystemExit("Install the agentic-policy module (real2sim2real[agentic]).") from error
        return agentic_main(argv[1:])
    parser = argparse.ArgumentParser(prog="r2s2r")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("modules", help="List selectable modules")
    sub.add_parser("pipelines", help="List named, composable pipeline recipes")
    sub.add_parser("agentic", help="Standalone automated or human-reviewed policy campaigns")
    p = sub.add_parser("assess", help="Check an explicit set of pinned evidence claims")
    p.add_argument("--config", required=True)
    p = sub.add_parser(
        "adopt", help="Wrap a saved control experiment as a verified artifact"
    )
    p.add_argument("--kind", choices=["grasp-trial", "arm-path-audit"], required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--output", required=True)
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
        elif args.command == "pipelines":
            result = PROFILES
        elif args.command == "assess":
            from .evidence_assessment import run

            result = {"output": str(run(args.config))}
        elif args.command == "adopt":
            from .control_artifacts import adopt

            result = {"output": str(adopt(args.source, args.output, args.kind))}
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
