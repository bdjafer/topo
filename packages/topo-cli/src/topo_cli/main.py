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

from topo_parser.graph import EdgeKind
from topo_parser.python import parse_python_project
from topo_analyzer.analysis import analyze
from topo_analyzer.layer_analysis import analyze_layer_signal
from topo_analyzer.projection import (
    AnalysisLevel,
    AnalysisPolicy,
    AnalysisProjectionConfig,
    discover_first_party_source_roots,
    load_analysis_policy,
)


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
    parser.add_argument(
        "--n-modules", type=int, default=None,
        help="Number of structural modules to detect (auto-detected if omitted)",
    )
    parser.add_argument(
        "--exclude", type=str, default=None,
        help="Comma-separated directory names to exclude (e.g. pycg,.venv,node_modules)",
    )
    parser.add_argument(
        "--scope",
        choices=["auto", "all", "first-party"],
        default=None,
        help="Analysis scope preset for monorepos and self-analysis",
    )
    parser.add_argument(
        "--level",
        choices=[level.value for level in AnalysisLevel],
        default=None,
        help="Analysis level (package, module, symbol)",
    )
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="Include low-level spectral diagnostics in text output",
    )
    parser.add_argument(
        "--layer-analysis",
        action="store_true",
        dest="layer_analysis",
        help="Analyze per-layer signal contribution instead of running the full pipeline",
    )

    args = parser.parse_args(argv)

    if not args.path.is_dir():
        print(f"Error: {args.path} is not a directory", file=sys.stderr)
        sys.exit(1)

    policy = _load_policy_or_exit(args.path)

    # Parse
    exclude_patterns = args.exclude.split(",") if args.exclude else None
    scope_roots = _resolve_scope_roots(args.path, args.scope, policy)
    graph = parse_python_project(
        args.path,
        exclude_patterns=exclude_patterns,
        include_roots=list(scope_roots) if scope_roots else None,
    )

    level = _resolve_analysis_level(args.level, policy)

    # Layer analysis mode
    if args.layer_analysis:
        layer_result = analyze_layer_signal(
            graph, level=level, scope_roots=scope_roots, n_modules=args.n_modules,
        )
        if args.as_json:
            print(json.dumps(_layer_analysis_to_dict(layer_result), indent=2))
        else:
            print(layer_result.summary())
        return

    # Analyze
    projection_config = AnalysisProjectionConfig.for_analysis(
        edge_kind=EdgeKind.CALLS if args.edge_kind == "combined" else EdgeKind(args.edge_kind),
        combined=args.edge_kind == "combined",
        level=level,
        scope_roots=scope_roots,
    )
    if args.edge_kind == "combined":
        result = analyze(
            graph,
            combined=True,
            n_modules=args.n_modules,
            projection_config=projection_config,
        )
    else:
        edge_kind = EdgeKind(args.edge_kind)
        result = analyze(
            graph,
            edge_kind=edge_kind,
            n_modules=args.n_modules,
            projection_config=projection_config,
        )

    # Output
    if args.as_json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(result.summary(verbose=args.diagnostics))

def _resolve_scope_roots(
    path: Path,
    scope: str | None,
    policy: AnalysisPolicy | None,
) -> tuple[Path, ...]:
    """Resolve CLI scope presets into concrete source roots."""
    scope_value = scope or (policy.scope if policy and policy.scope else "auto")
    if scope_value == "all":
        return ()
    first_party_roots = discover_first_party_source_roots(path)
    if scope_value == "first-party":
        return first_party_roots
    return first_party_roots


def _resolve_analysis_level(
    level: str | None,
    policy: AnalysisPolicy | None,
) -> AnalysisLevel:
    """Resolve CLI and policy defaults for analysis level."""
    if level is not None:
        return AnalysisLevel(level)
    if policy and policy.level is not None:
        return policy.level
    return AnalysisLevel.MODULE
def _layer_analysis_to_dict(result) -> dict:
    """Serialize a LayerAnalysisResult to a JSON-friendly dict."""
    def _signal_dict(signal):
        return {
            "label": signal.label,
            "weights": {k.value: v for k, v in signal.weights.items()},
            "nmi": round(signal.nmi, 4) if signal.nmi is not None else None,
            "silhouette": round(signal.silhouette, 4) if signal.silhouette is not None else None,
            "quality_score": round(signal.quality_score, 4),
            "n_modules": signal.n_modules,
            "coverage_ratio": round(signal.coverage_ratio, 4),
            "edge_count": signal.edge_count,
        }

    return {
        "signals": [_signal_dict(s) for s in result.signals],
        "best_single": _signal_dict(result.best_single) if result.best_single else None,
        "best_combined": _signal_dict(result.best_combined) if result.best_combined else None,
        "recommended_weights": {
            k.value: v for k, v in result.recommended_weights.items()
        },
    }


def _load_policy_or_exit(path: Path) -> AnalysisPolicy | None:
    """Load repo policy and turn parse errors into CLI-friendly failures."""
    try:
        return load_analysis_policy(path)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
