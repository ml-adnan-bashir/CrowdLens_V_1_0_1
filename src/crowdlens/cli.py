"""CLI entry point; no input video is uploaded to any service."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from .config import Settings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="crowdlens", description="Human-reviewed local video analysis")
    sub = parser.add_subparsers(dest="command", required=True)
    analyze_cmd = sub.add_parser("analyze", help="Analyze one local video")
    analyze_cmd.add_argument("video", type=Path, help="Input MP4/other OpenCV-supported video")
    analyze_cmd.add_argument("--config", type=Path, default=None, help="YAML settings path")
    analyze_cmd.add_argument("--output", type=Path, default=None, help="NEW output directory; no overwrites")
    analyze_cmd.add_argument("--weapon-model", type=str, default=None, help="Path to licensed YOLO weapon-detection weights")
    analyze_cmd.add_argument("--person-model", type=str, default=None, help="YOLO model containing 'person' class")
    analyze_cmd.add_argument("--reid-backend", choices=["histogram", "resnet18"], default=None)
    analyze_cmd.add_argument("--reid-similarity", type=float, default=None)
    check = sub.add_parser("check-config", help="Validate YAML and print effective settings")
    check.add_argument("--config", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        config = Settings.load(args.config)
        if args.command == "check-config":
            print(json.dumps(config.as_dict(), indent=2))
            return 0
        if args.weapon_model:
            config.weapon_model = args.weapon_model
        if args.person_model:
            config.person_model = args.person_model
        if args.reid_backend:
            config.reid_backend = args.reid_backend
        if args.reid_similarity is not None:
            config.reid_similarity = args.reid_similarity
        config.validate()
        from .pipeline import analyze

        output = args.output or Path("runs") / datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ_%f")
        result = analyze(args.video, output, config)
        print(json.dumps({"output": str(output.resolve()), "frames": result["frames"],
                          "events": result["events_by_type"],
                          "weapon_detection_enabled": result["weapon_detection_enabled"]}, indent=2))
        return 0
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as exc:
        print(f"crowdlens: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

