"""Standalone CLI entry point for topo-parser.

Usage:
    python -m topo_parser <path> [--exclude <dirs>] [--scope <preset>]

Outputs graph.schema.json to stdout.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from topo_parser.python import parse_python_project


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="topo_parser",
        description="Parse a Python project into CodeGraph JSON (graph.schema.json)",
    )
    parser.add_argument("path", type=Path, help="Path to Python project root")
    parser.add_argument(
        "--exclude",
        type=str,
        default=None,
        help="Comma-separated directory names to exclude",
    )
    parser.add_argument(
        "--scope",
        choices=["auto", "all", "first-party"],
        default=None,
        help="Scope preset (currently unused by parser, reserved for future use)",
    )
    args = parser.parse_args()

    if not args.path.is_dir():
        print(f"Error: {args.path} is not a directory", file=sys.stderr)
        sys.exit(1)

    exclude_patterns = args.exclude.split(",") if args.exclude else None

    graph = parse_python_project(
        args.path,
        exclude_patterns=exclude_patterns,
    )

    json.dump(graph.to_dict(), sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
