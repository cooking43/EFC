#!/usr/bin/env python3
"""Install the lightweight EFC Unity overlay into a clean Gewu checkout."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy EFC's Unity files into a downloaded Unity-RL-Playground checkout."
    )
    parser.add_argument(
        "gewu_repository",
        type=Path,
        help="Path to the root of loongOpen/Unity-RL-Playground",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and list changes without copying files",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repository = args.gewu_repository.expanduser().resolve()
    unity_project = repository / "gewu"
    marker = unity_project / "Assets" / "Playground" / "RobotRLAgent.cs"

    if not marker.is_file():
        print(
            "error: the target does not look like a Unity-RL-Playground checkout; "
            f"missing {marker}",
            file=sys.stderr,
        )
        return 2

    release_root = Path(__file__).resolve().parents[1]
    overlay = release_root / "unity"
    if not overlay.is_dir():
        print(f"error: overlay directory is missing: {overlay}", file=sys.stderr)
        return 2

    files = sorted(path for path in overlay.rglob("*") if path.is_file())
    print(f"EFC overlay: {overlay}")
    print(f"Gewu target: {unity_project}")
    print(f"Files: {len(files)}")

    for source in files:
        relative = source.relative_to(overlay)
        destination = unity_project / relative
        action = "replace" if destination.exists() else "add"
        print(f"[{action}] {relative.as_posix()}")
        if not args.dry_run:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)

    if args.dry_run:
        print("Dry run complete; no files were changed.")
    else:
        print("Overlay installed. Review the target checkout with 'git diff'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
