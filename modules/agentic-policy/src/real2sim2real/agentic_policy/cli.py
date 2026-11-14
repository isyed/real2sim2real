"""Standalone campaign command line; no model call occurs during init/status."""

import argparse
import json

from . import campaign


def main(argv=None):
    parser = argparse.ArgumentParser(prog="r2s2r agentic")
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init")
    init.add_argument("--bundle", required=True)
    init.add_argument("--output", required=True)
    for name in ["advance", "status", "retry", "record-clock-rejection"]:
        p = sub.add_parser(name)
        p.add_argument("--campaign", required=True)
    submit = sub.add_parser("submit")
    submit.add_argument("--campaign", required=True)
    submit.add_argument("--candidate", required=True)
    export = sub.add_parser("export")
    export.add_argument("--campaign", required=True)
    export.add_argument("--proposal", required=True)
    export.add_argument("--output", required=True)
    preview = sub.add_parser("preview-author")
    preview.add_argument("--campaign", required=True)
    preview.add_argument("--output", required=True)
    extend = sub.add_parser("extend")
    extend.add_argument("--campaign", required=True)
    extend.add_argument("--additional-revisions", type=int, required=True)
    extend.add_argument("--reason", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "init":
            result = campaign.initialize(args.bundle, args.output)
        elif args.command == "preview-author":
            result = campaign.preview_author(args.campaign, args.output)
        elif args.command == "export":
            result = campaign.export_policy(args.campaign, args.proposal, args.output)
        elif args.command == "submit":
            result = campaign.submit(args.campaign, args.candidate)
        elif args.command == "extend":
            from .extensions import extend_budget

            result = extend_budget(
                args.campaign, args.additional_revisions, args.reason
            )
        elif args.command == "record-clock-rejection":
            from .recovery import record_clock_rejection

            result = record_clock_rejection(args.campaign)
        else:
            result = getattr(campaign, args.command)(args.campaign)
        print(json.dumps(result, indent=2, allow_nan=False))
    except (ValueError, OSError, RuntimeError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
