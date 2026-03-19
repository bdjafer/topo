"""
CLI entry point for topo.

Usage:
    topo <path>                       Analyze a Python project (parse + analyze).
    topo parse <path> -o graph.json   Parse only, output the graph as JSON.
    topo analyze --input graph.json   Analyze a pre-parsed graph.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from topo_parser.graph import CodeGraph, EdgeKind
from topo_parser.python import parse_python_project
from topo_analyzer.analysis import analyze
from topo_analyzer.projection import (
    AnalysisLevel,
    AnalysisPolicy,
    AnalysisProjectionConfig,
    discover_first_party_source_roots,
    load_analysis_policy,
)

_SUBCOMMANDS = {"parse", "analyze"}


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]

    # Route to subcommand or default (backward-compatible) mode.
    if argv and argv[0] in _SUBCOMMANDS:
        command = argv[0]
        rest = argv[1:]
        if command == "parse":
            _cmd_parse(rest)
        else:
            _cmd_analyze(rest)
    else:
        _cmd_default(argv)


def _cmd_parse(argv: list[str]) -> None:
    """Parse a project and output CodeGraph JSON."""
    parser = argparse.ArgumentParser(prog="topo parse", description="Parse a Python project into CodeGraph JSON")
    parser.add_argument("path", type=Path, help="Path to Python project root")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output file (default: stdout)")
    parser.add_argument("--exclude", type=str, default=None, help="Comma-separated directory names to exclude")
    parser.add_argument(
        "--scope", choices=["auto", "all", "first-party"], default=None,
        help="Analysis scope preset for monorepos",
    )
    args = parser.parse_args(argv)

    if not args.path.is_dir():
        print(f"Error: {args.path} is not a directory", file=sys.stderr)
        sys.exit(1)

    policy = _load_policy_or_exit(args.path)
    exclude_patterns = args.exclude.split(",") if args.exclude else None
    scope_roots = _resolve_scope_roots(args.path, args.scope, policy)

    graph = parse_python_project(
        args.path,
        exclude_patterns=exclude_patterns,
        include_roots=list(scope_roots) if scope_roots else None,
    )

    output = json.dumps(graph.to_dict(), indent=2)
    if args.output:
        Path(args.output).write_text(output + "\n")
        print(f"Wrote {graph.node_count} nodes, {graph.edge_count} edges to {args.output}", file=sys.stderr)
    else:
        print(output)


def _cmd_analyze(argv: list[str]) -> None:
    """Analyze a pre-parsed CodeGraph JSON."""
    parser = argparse.ArgumentParser(prog="topo analyze", description="Analyze a pre-parsed CodeGraph JSON")
    parser.add_argument("--input", type=str, required=True, help="Path to CodeGraph JSON file")
    _add_analysis_args(parser)
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    if not input_path.is_file():
        print(f"Error: {args.input} is not a file", file=sys.stderr)
        sys.exit(1)

    graph = CodeGraph.from_dict(json.loads(input_path.read_text()))
    _run_analysis(graph, args, project_root=None)


def _cmd_default(argv: list[str]) -> None:
    """Default command: parse + analyze (backward-compatible)."""
    parser = argparse.ArgumentParser(
        prog="topo",
        description="Structural intelligence for codebases",
    )
    parser.add_argument("path", type=Path, help="Path to Python project root")
    _add_analysis_args(parser)
    parser.add_argument("--exclude", type=str, default=None, help="Comma-separated directory names to exclude")
    parser.add_argument(
        "--scope", choices=["auto", "all", "first-party"], default=None,
        help="Analysis scope preset for monorepos",
    )
    args = parser.parse_args(argv)

    if not args.path.is_dir():
        print(f"Error: {args.path} is not a directory", file=sys.stderr)
        sys.exit(1)

    policy = _load_policy_or_exit(args.path)
    exclude_patterns = args.exclude.split(",") if args.exclude else None
    scope_roots = _resolve_scope_roots(args.path, args.scope, policy)

    graph = parse_python_project(
        args.path,
        exclude_patterns=exclude_patterns,
        include_roots=list(scope_roots) if scope_roots else None,
    )

    _run_analysis(graph, args, project_root=args.path.resolve(), policy=policy, scope_roots=scope_roots)


def _add_analysis_args(parser: argparse.ArgumentParser) -> None:
    """Add shared analysis flags to a parser."""
    parser.add_argument("--json", action="store_true", dest="as_json", help="Output as JSON")
    parser.add_argument(
        "--edge-kind", default="combined",
        help="Edge layer to analyze (calls, imports, inherits, contains, combined)",
    )
    parser.add_argument("--n-modules", type=int, default=None, help="Number of modules (auto if omitted)")
    parser.add_argument(
        "--level", choices=[level.value for level in AnalysisLevel], default=None,
        help="Analysis level (package, module, symbol)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Show full details")
    parser.add_argument("--diagnostics", action="store_true", help="Show spectral diagnostics")
    parser.add_argument("--no-color", action="store_true", help="Disable colored output")
    parser.add_argument(
        "--backend", choices=["auto", "rust"], default="auto",
        help="Compute backend (auto=scipy/numpy, rust=topo-core Rust engine)",
    )


def _run_analysis(
    graph: CodeGraph,
    args: argparse.Namespace,
    *,
    project_root: Path | None = None,
    policy: AnalysisPolicy | None = None,
    scope_roots: tuple[Path, ...] = (),
) -> None:
    """Shared analysis + output logic."""
    import os
    if getattr(args, "backend", "auto") == "rust":
        os.environ["TOPO_BACKEND"] = "rust"
    level = _resolve_analysis_level(getattr(args, "level", None), policy)
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

    if args.as_json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        ignores = policy.ignores if policy else {}
        use_color = not args.no_color and sys.stdout.isatty()
        print(result.summary(
            verbose=args.verbose,
            diagnostics=args.diagnostics,
            ignores=ignores,
            project_root=project_root,
            color=use_color,
        ))


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


def _load_policy_or_exit(path: Path) -> AnalysisPolicy | None:
    """Load repo policy and turn parse errors into CLI-friendly failures."""
    try:
        return load_analysis_policy(path)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
