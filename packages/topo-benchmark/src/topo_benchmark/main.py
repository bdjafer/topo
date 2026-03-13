"""CLI entry point for topo-benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def cmd_run(args: argparse.Namespace) -> None:
    """Run a benchmark."""
    from topo_benchmark.runner import run_benchmark

    output_dir = Path(args.output_dir) if args.output_dir else None
    dataset_root = Path(args.dataset_root) if args.dataset_root else None

    scorecard = run_benchmark(
        tier=args.tier,
        split=args.split,
        dataset_root=dataset_root,
        output_dir=output_dir,
    )

    print(f"Overall primary score: {scorecard.overall_primary:.4f}")
    print(f"Promotion decision: {scorecard.promotion_decision}")
    for dim, score in scorecard.dimensions.items():
        print(f"  {dim}: {score:.4f}")

    if output_dir:
        print(f"\nArtifacts written to: {output_dir}")


def cmd_compare(args: argparse.Namespace) -> None:
    """Compare two benchmark runs."""
    from topo_benchmark.compare import compare_runs

    result = compare_runs(
        candidate_dir=Path(args.candidate),
        reference_dir=Path(args.reference),
    )

    print(f"Overall delta: {result['overall_delta']:+.4f}")
    print(f"Promotion: {result['promotion_decision']}")
    print()
    for dim, d in result["dimensions"].items():
        marker = "!" if d["regressed"] else " "
        print(f"  {marker} {dim}: {d['delta']:+.4f} [{d['ci_low']:+.4f}, {d['ci_high']:+.4f}]")

    if args.fail_on_regression and result["promotion_decision"] != "pass":
        sys.exit(1)


def cmd_report(args: argparse.Namespace) -> None:
    """Regenerate a summary report."""
    from topo_benchmark.report import generate_summary
    from topo_benchmark.scorecard import Scorecard

    run_dir = Path(args.input)
    scorecard = Scorecard.load(run_dir / "scorecard.json")

    dimensions = {}
    dim_path = run_dir / "dimensions.json"
    if dim_path.exists():
        dimensions = json.loads(dim_path.read_text())

    baselines = {}
    baselines_dir = run_dir / "baselines"
    if baselines_dir.is_dir():
        for bp in baselines_dir.glob("*.json"):
            baselines[bp.stem] = json.loads(bp.read_text())

    summary = generate_summary(scorecard.to_dict(), dimensions, baselines)
    output_path = run_dir / "summary.md"
    output_path.write_text(summary)
    print(summary)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="topo-benchmark",
        description="Benchmark harness for topo-analyzer",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # run
    run_parser = subparsers.add_parser("run", help="Run a benchmark")
    run_parser.add_argument("--tier", default="analyzer", choices=["analyzer", "e2e"])
    run_parser.add_argument("--split", default="public", choices=["public", "hidden", "smoke"])
    run_parser.add_argument("--output-dir", default=None)
    run_parser.add_argument("--dataset-root", default=None)
    run_parser.set_defaults(func=cmd_run)

    # compare
    cmp_parser = subparsers.add_parser("compare", help="Compare two benchmark runs")
    cmp_parser.add_argument("--candidate", required=True)
    cmp_parser.add_argument("--reference", required=True)
    cmp_parser.add_argument("--fail-on-regression", action="store_true")
    cmp_parser.set_defaults(func=cmd_compare)

    # report
    rpt_parser = subparsers.add_parser("report", help="Regenerate summary report")
    rpt_parser.add_argument("--input", required=True)
    rpt_parser.set_defaults(func=cmd_report)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
