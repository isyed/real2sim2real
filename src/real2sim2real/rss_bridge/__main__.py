"""CLI: python -m real2sim2real.rss_bridge {export,import,check-camera}."""

import argparse
import json
from pathlib import Path

from real2sim2real.artifacts import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export")
    for name in ("geometry", "mesh", "human", "output", "source-frame",
                 "human-coordinate-frame", "rss-revision", "transform-json"):
        export.add_argument("--" + name, required=True)
    export.add_argument("--width", type=int, required=True)
    export.add_argument("--height", type=int, required=True)
    export.add_argument("--rgb")
    edit = sub.add_parser("import")
    for name in ("bundle", "poses", "output"):
        edit.add_argument("--" + name, required=True)
    check = sub.add_parser("check-camera")
    check.add_argument("--bundle", required=True)
    check.add_argument("--frames", nargs="+", type=int, required=True)
    check.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "export":
        from .bundle import create_bundle
        result = create_bundle(
            args.geometry, args.mesh, args.human, args.output,
            source_to_rss=json.loads(Path(args.transform_json).read_text()),
            width=args.width, height=args.height, source_frame=args.source_frame,
            human_coordinate_frame=args.human_coordinate_frame,
            rss_revision=args.rss_revision, rgb=args.rgb)
        print(result)
    elif args.command == "import":
        from .bundle import import_poses
        print(import_poses(args.bundle, args.poses, args.output))
    else:
        from .scene import camera_parity
        result = camera_parity(args.bundle, args.frames)
        write_json(args.output, result)
        if not result["passed"]:
            raise SystemExit("RSS camera parity failed")
        print(args.output)


if __name__ == "__main__":
    main()
