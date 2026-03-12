"""
CLI entry point for topo.

Usage:
    topo <path>          Analyze a Python project and print structural report.
    topo <path> --json   Output analysis as JSON (for LLM context).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from topo_parser.python import parse_python_project
from topo_analyzer.analysis import analyze


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="topo",
        description="Structural intelligence for codebases",
    )
    parser.add_argument("path", type=Path, help="Path to Python project root")
    parser.add_argument("--json", action="store_true", dest="as_json", help="Output as JSON")
    parser.add_argument(
        "--edge-kind", default="combined",
        help="Edge layer to analyze (calls, imports, inherits, contains, combined)",
    )

    args = parser.parse_args(argv)

    if not args.path.is_dir():
        print(f"Error: {args.path} is not a directory", file=sys.stderr)
        sys.exit(1)

    # Parse
    graph = parse_python_project(args.path)

    # Analyze
    from topo_parser.graph import EdgeKind
    if args.edge_kind == "combined":
        result = analyze(graph, combined=True)
    else:
        edge_kind = EdgeKind(args.edge_kind)
        result = analyze(graph, edge_kind=edge_kind)

    # Output
    if args.as_json:
        print(json.dumps(_to_dict(result), indent=2))
    else:
        print(result.summary())


def _to_dict(result) -> dict:
    """Convert analysis result to a JSON-serializable dict."""
    return {
        "graph": {
            "nodes": result.graph.node_count,
            "edges": result.graph.edge_count,
        },
        "spectral": {
            "fiedler_value": result.spectral.fiedler_value,
            "eigenvalues": result.spectral.eigenvalues.tolist(),
        } if result.spectral else None,
        "modules": [
            {"id": m.id, "size": m.size, "members": m.node_ids}
            for m in result.modules
        ],
        "roles": [
            {
                "node_id": r.node_id,
                "role": r.role.value,
                "degree": r.degree,
                "betweenness": round(r.betweenness, 4),
            }
            for r in result.roles
        ],
        "anomalies": [
            {
                "kind": a.kind.value,
                "node_ids": a.node_ids,
                "description": a.description,
                "severity": round(a.severity, 2),
            }
            for a in result.anomalies
        ],
    }


if __name__ == "__main__":
    main()
