#!/usr/bin/env python3
"""
Validate that all committed example artifacts are consistent.

Runs in CI without cloning external repos — only checks committed files.

Usage:
    python3 examples/scripts/validate_examples.py

Exit code 0 = all valid, 1 = errors found.
"""

import json
import sys
import tomllib
from pathlib import Path


def main() -> int:
    script_dir = Path(__file__).parent
    examples_dir = script_dir.parent
    registry_path = examples_dir / "registry.toml"

    if not registry_path.exists():
        print(f"ERROR: registry.toml not found at {registry_path}")
        return 1

    with open(registry_path, "rb") as f:
        registry = tomllib.load(f)

    examples = registry.get("example", [])
    if not examples:
        print("WARNING: No examples in registry.toml")
        return 0

    errors: list[str] = []
    warnings: list[str] = []

    for ex in examples:
        name = ex["name"]
        ex_dir = examples_dir / name

        if not ex_dir.exists():
            warnings.append(f"{name}: directory does not exist (not yet generated?)")
            continue

        # Check required artifacts
        for artifact in ["graph.json", "analysis.json", "analysis.txt"]:
            path = ex_dir / artifact
            if not path.exists():
                errors.append(f"{name}: missing {artifact}")
            elif path.stat().st_size == 0:
                errors.append(f"{name}: {artifact} is empty")

        # Validate graph.json structure
        graph_path = ex_dir / "graph.json"
        if graph_path.exists():
            try:
                with open(graph_path) as f:
                    graph = json.load(f)
                if "nodes" not in graph:
                    errors.append(f"{name}: graph.json missing 'nodes' key")
                if "edges" not in graph:
                    errors.append(f"{name}: graph.json missing 'edges' key")
            except json.JSONDecodeError as e:
                errors.append(f"{name}: graph.json invalid JSON: {e}")

        # Validate analysis.json structure
        analysis_path = ex_dir / "analysis.json"
        if analysis_path.exists():
            try:
                with open(analysis_path) as f:
                    analysis = json.load(f)
                for key in ["architecture", "issues", "roles"]:
                    if key not in analysis:
                        errors.append(f"{name}: analysis.json missing '{key}' key")
            except json.JSONDecodeError as e:
                errors.append(f"{name}: analysis.json invalid JSON: {e}")

        # Validate metadata.json if present
        meta_path = ex_dir / "metadata.json"
        if meta_path.exists():
            try:
                with open(meta_path) as f:
                    meta = json.load(f)

                # Cross-check graph stats
                graph_meta = meta.get("graph", {})
                if graph_path.exists() and graph_meta:
                    with open(graph_path) as f:
                        graph = json.load(f)
                    actual_nodes = len(graph.get("nodes", []))
                    if graph_meta.get("nodes") != actual_nodes:
                        warnings.append(
                            f"{name}: metadata node count ({graph_meta.get('nodes')}) "
                            f"!= graph.json ({actual_nodes})"
                        )
            except json.JSONDecodeError as e:
                errors.append(f"{name}: metadata.json invalid JSON: {e}")

        # Check required registry fields
        for field in ["repo", "commit", "language", "entrypoint"]:
            if field not in ex:
                errors.append(f"{name}: registry entry missing '{field}'")

    # Report
    total = len(examples)
    generated = sum(1 for ex in examples if (examples_dir / ex["name"]).exists())

    print(f"Registry: {total} examples, {generated} generated")
    print()

    if warnings:
        print(f"Warnings ({len(warnings)}):")
        for w in warnings:
            print(f"  ⚠ {w}")
        print()

    if errors:
        print(f"Errors ({len(errors)}):")
        for e in errors:
            print(f"  ✗ {e}")
        return 1

    print("All generated examples valid.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
