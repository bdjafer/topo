#!/usr/bin/env python3
"""Run Topo Health Score computation on dataset splits.

Mirrors the evaluation pipeline pattern: reads split files, saves results
into the checkpoint directory alongside evaluation_report.json.

Usage:
    python -m topo_eval.scripts.run_evaluation \
        --checkpoint checkpoints/rgin_v2 \
        --data-dir examples \
        --split-dir examples/splits

    # Specific split
    python -m scripts.run_health \
        --checkpoint checkpoints/rgin_v2 \
        --data-dir examples \
        --split-dir examples/splits \
        --eval-split val
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description="Compute THS for dataset splits")
    parser.add_argument(
        "--checkpoint", type=Path, required=True,
        help="Model checkpoint directory (e.g., checkpoints/rgin_v2)",
    )
    parser.add_argument(
        "--data-dir", type=Path, required=True,
        help="Root directory containing repo feature dirs (e.g., examples/)",
    )
    parser.add_argument(
        "--split-dir", type=Path, required=True,
        help="Directory with train.txt, val.txt, test.txt",
    )
    parser.add_argument(
        "--eval-split", type=str, default="all",
        choices=["train", "val", "test", "all"],
        help="Which split(s) to evaluate (default: all)",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.7,
        help="Coherence weight α (default: 0.7)",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device: cpu, cuda, mps, or auto",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run even if health_report.json exists",
    )
    args = parser.parse_args()

    # Resolve paths relative to project root
    topo_root = Path(__file__).resolve().parent.parent.parent.parent
    checkpoint_dir = args.checkpoint if args.checkpoint.is_absolute() else topo_root / args.checkpoint
    data_dir = args.data_dir if args.data_dir.is_absolute() else topo_root / args.data_dir
    split_dir = args.split_dir if args.split_dir.is_absolute() else topo_root / args.split_dir

    # Skip if already computed
    report_path = checkpoint_dir / "health_report.json"
    report_txt_path = checkpoint_dir / "health_report.txt"
    if report_path.exists() and not args.force:
        print(f"Health report already exists at {report_path}")
        print("Use --force to re-run.")
        if report_txt_path.exists():
            print("\n" + report_txt_path.read_text())
        return 0

    # Resolve device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print(f"Device: {device}", file=sys.stderr)

    # Add packages to path
    sys.path.insert(0, str(topo_root / "packages" / "topo-dataset" / "src"))
    sys.path.insert(0, str(topo_root / "packages" / "topo-model" / "src"))

    from topo_dataset.loader import load_graph
    from topo_model.health import load_model, compute_health

    # Load model
    print(f"\n[1/3] Loading model from {checkpoint_dir}...", file=sys.stderr)
    model = load_model(checkpoint_dir / "best_model.pt", device=device)
    print("Model loaded.", file=sys.stderr)

    # Load model metadata
    model_info = {}
    metadata_path = checkpoint_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            model_info = json.load(f)

    # Determine splits
    if args.eval_split == "all":
        split_names = ["train", "val", "test"]
    else:
        split_names = [args.eval_split]

    # Load repos per split
    print(f"\n[2/3] Loading dataset splits...", file=sys.stderr)
    split_repos: dict[str, list[str]] = {}
    for split in split_names:
        split_file = split_dir / f"{split}.txt"
        if not split_file.exists():
            print(f"  Warning: {split_file} not found, skipping", file=sys.stderr)
            continue
        repos = [r.strip() for r in split_file.read_text().strip().split("\n") if r.strip()]
        # Filter to repos that have features
        repos = [r for r in repos if (data_dir / r / "features.npz").exists() and (data_dir / r / "graph.json").exists()]
        split_repos[split] = repos
        print(f"  {split}: {len(repos)} repos ({', '.join(repos)})", file=sys.stderr)

    total_repos = sum(len(r) for r in split_repos.values())
    print(f"  Total: {total_repos} repos", file=sys.stderr)

    # Process each split
    print(f"\n[3/3] Computing health scores...", file=sys.stderr)
    t0 = time.time()
    results: dict[str, dict] = {}

    for split, repos in split_repos.items():
        print(f"\n  --- {split} split ---", file=sys.stderr)
        for name in repos:
            repo_dir = data_dir / name
            t_repo = time.time()

            try:
                data = load_graph(repo_dir)
                with open(repo_dir / "graph.json") as f:
                    graph = json.load(f)

                health = compute_health(model, data, graph, alpha=args.alpha, device=device)
                elapsed = time.time() - t_repo

                entry = health.to_dict()
                entry["split"] = split
                entry["time_s"] = round(elapsed, 2)
                results[name] = entry

                _print_health_line(name, health, elapsed, split)

            except Exception as e:
                print(f"  ERROR [{name}]: {e}", file=sys.stderr)
                import traceback
                traceback.print_exc(file=sys.stderr)
                results[name] = {"error": str(e), "split": split}

    total_time = time.time() - t0

    # Build report
    report = {
        "alpha": args.alpha,
        "checkpoint": str(checkpoint_dir.resolve()),
        "device": str(device),
        "eval_split": args.eval_split,
        "n_repos": total_repos,
        "total_time_s": round(total_time, 1),
        "model_info": model_info,
        "splits": {s: repos for s, repos in split_repos.items()},
        "results": results,
    }

    # Save JSON report
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nJSON report saved to {report_path}", file=sys.stderr)

    # Generate and save text report
    txt = _format_text_report(report, split_repos, results)
    with open(report_txt_path, "w") as f:
        f.write(txt)
    print(f"Text report saved to {report_txt_path}", file=sys.stderr)
    print("\n" + txt)

    return 0


def _print_health_line(name: str, health, elapsed: float, split: str):
    """Print a single health result line."""
    bar_c = _bar(health.coherence)
    bar_f = _bar(health.flow)
    print(
        f"  {name:<20s} THS={health.topo_health_score:.4f}  "
        f"coh={health.coherence:.4f} {bar_c}  "
        f"flow={health.flow:.4f} {bar_f}  "
        f"({health.n_nodes:>5d} nodes, {elapsed:.1f}s)",
        file=sys.stderr,
    )


def _bar(value: float, width: int = 10) -> str:
    """Generate a progress bar string."""
    clamped = max(0.0, min(1.0, value))
    filled = int(round(clamped * width))
    return "\u2588" * filled + "\u2591" * (width - filled)


def _format_text_report(report: dict, split_repos: dict, results: dict) -> str:
    """Generate a formatted text report mirroring evaluation_report.txt."""
    lines = []
    lines.append("=" * 70)
    lines.append("TOPO HEALTH SCORE REPORT")
    lines.append("=" * 70)
    lines.append(f"Checkpoint: {report['checkpoint']}")
    lines.append(f"Alpha (coherence weight): {report['alpha']}")
    lines.append(f"Device: {report['device']}")
    lines.append(f"Total time: {report['total_time_s']}s")

    model_info = report.get("model_info", {})
    if model_info:
        lines.append(f"Model: {model_info.get('architecture', '?')} "
                      f"(hidden={model_info.get('hidden_dim', '?')}, "
                      f"layers={model_info.get('n_layers', '?')}, "
                      f"params={model_info.get('n_params', '?'):,})")

    # Per-split tables
    for split, repos in split_repos.items():
        lines.append("")
        lines.append("-" * 70)
        lines.append(f"{split.upper()} SPLIT ({len(repos)} repos)")
        lines.append("-" * 70)
        lines.append(f"  {'Repo':<20s} {'THS':>6s} {'Coher':>6s} {'Flow':>6s} "
                      f"{'CycFr':>6s} {'LayCf':>6s} {'MedErr':>7s} {'Nodes':>6s}")
        lines.append("  " + "-" * 65)

        split_results = []
        for name in repos:
            r = results.get(name, {})
            if "error" in r:
                lines.append(f"  {name:<20s} ERROR: {r['error']}")
                continue
            lines.append(
                f"  {name:<20s} {r['topo_health_score']:>6.4f} {r['coherence']:>6.4f} "
                f"{r['flow']:>6.4f} {r['cycle_freedom']:>6.4f} {r['layer_conformance']:>6.4f} "
                f"{r['median_reconstruction_error']:>7.4f} {r['n_nodes']:>6d}"
            )
            split_results.append(r)

        # Split averages
        if split_results:
            avg_ths = np.mean([r["topo_health_score"] for r in split_results])
            avg_coh = np.mean([r["coherence"] for r in split_results])
            avg_flow = np.mean([r["flow"] for r in split_results])
            lines.append("  " + "-" * 65)
            lines.append(f"  {'AVG':<20s} {avg_ths:>6.4f} {avg_coh:>6.4f} {avg_flow:>6.4f}")

    # Overall summary
    valid = [r for r in results.values() if "error" not in r]
    if valid:
        lines.append("")
        lines.append("=" * 70)
        lines.append("OVERALL SUMMARY")
        lines.append("=" * 70)
        all_ths = [r["topo_health_score"] for r in valid]
        all_coh = [r["coherence"] for r in valid]
        all_flow = [r["flow"] for r in valid]
        lines.append(f"  THS:       {np.mean(all_ths):.4f} +/- {np.std(all_ths):.4f}  "
                      f"(min={np.min(all_ths):.4f}, max={np.max(all_ths):.4f})")
        lines.append(f"  Coherence: {np.mean(all_coh):.4f} +/- {np.std(all_coh):.4f}  "
                      f"(min={np.min(all_coh):.4f}, max={np.max(all_coh):.4f})")
        lines.append(f"  Flow:      {np.mean(all_flow):.4f} +/- {np.std(all_flow):.4f}  "
                      f"(min={np.min(all_flow):.4f}, max={np.max(all_flow):.4f})")

        # Variance warning
        ths_range = np.max(all_ths) - np.min(all_ths)
        if ths_range < 0.10:
            lines.append("")
            lines.append(f"  WARNING: Low THS variance (range={ths_range:.4f}). "
                          "Score may lack discriminative power.")
            lines.append("  This is expected with a minimal (not-scaled) model + unmasked inference.")

    lines.append("=" * 70)
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.exit(main())
