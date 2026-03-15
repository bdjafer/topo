"""CLI entry point for experiments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from topo_benchmark.experiments.exp1_architecture import run_experiment_1
from topo_benchmark.experiments.report import format_exp1_report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run pre-registered experiments to prove/disprove the core bet.",
    )
    parser.add_argument(
        "--experiment",
        type=int,
        choices=[1, 2, 3, 4],
        default=1,
        help="Which experiment to run (default: 1)",
    )
    parser.add_argument(
        "--codebases-root",
        type=Path,
        default=Path("/tmp/topo-experiment-codebases"),
        help="Directory to clone/find codebases",
    )
    parser.add_argument(
        "--labels-root",
        type=Path,
        default=Path("benchmark/gold_labels"),
        help="Directory containing gold labels",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark/results"),
        help="Where to write results",
    )
    parser.add_argument(
        "--codebases",
        nargs="*",
        help="Only run these codebases (default: all registered)",
    )

    args = parser.parse_args()

    if args.experiment == 1:
        result = run_experiment_1(
            codebases_root=args.codebases_root,
            labels_root=args.labels_root,
            output_dir=args.output_dir,
            codebase_filter=args.codebases,
        )
        print(format_exp1_report(result))
        if result.verdict == "FAIL":
            sys.exit(1)
    else:
        print(f"Experiment {args.experiment} not yet implemented.")
        sys.exit(2)


if __name__ == "__main__":
    main()
