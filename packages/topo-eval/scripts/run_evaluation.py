#!/usr/bin/env python3
"""Full evaluation pipeline for R-GIN model.

Usage:
    python -m topo_eval.scripts.run_evaluation \
        --checkpoint checkpoints/rgin_v2 \
        --data-dir examples \
        --split-dir examples/splits \
        --output-dir eval_results
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

from topo_model.config import RGINConfig
from topo_model.rgin import RGIN
from topo_dataset.loader import load_graph

from topo_eval.tier1 import tier1_intrinsic_metrics
from topo_eval.tier2 import tier2_phase2_agreement
from topo_eval.tier3 import tier3_perturbation_test
from topo_eval.tier4 import tier4_structural_consistency
from topo_eval.baselines import run_baselines
from topo_eval.gate import go_no_go_gate
from topo_eval.report import generate_report


def load_model(checkpoint_dir: Path, device: torch.device) -> RGIN:
    """Load a trained R-GIN model from a checkpoint directory."""
    config_path = checkpoint_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config_data = json.load(f)
        config = RGINConfig(**{
            k: v for k, v in config_data.items()
            if k in RGINConfig.__dataclass_fields__
        })
    else:
        config = RGINConfig()

    model = RGIN(config)

    # Try best_model.pt first, then final_model.pt
    for name in ["best_model.pt", "final_model.pt"]:
        ckpt_path = checkpoint_dir / name
        if ckpt_path.exists():
            state = torch.load(ckpt_path, map_location=device, weights_only=False)
            if "model_state" in state:
                model.load_state_dict(state["model_state"])
            elif "model_state_dict" in state:
                model.load_state_dict(state["model_state_dict"])
            else:
                model.load_state_dict(state)
            print(f"Loaded model from {ckpt_path}")
            break
    else:
        raise FileNotFoundError(f"No model checkpoint found in {checkpoint_dir}")

    model.to(device)
    model.eval()
    return model


def load_dataset(data_dir: Path, split_file: Path) -> list:
    """Load graphs listed in a split file."""
    repos = split_file.read_text().strip().split("\n")
    repos = [r.strip() for r in repos if r.strip()]

    dataset = []
    for repo in repos:
        repo_dir = data_dir / repo
        npz_path = repo_dir / "features.npz"
        if not npz_path.exists():
            print(f"  Skipping {repo}: no features.npz")
            continue
        data = load_graph(repo_dir)
        dataset.append(data)
        print(f"  Loaded {repo}: {data['node'].num_nodes} nodes")

    return dataset


def main():
    parser = argparse.ArgumentParser(description="R-GIN Evaluation Pipeline")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Model checkpoint directory")
    parser.add_argument("--data-dir", type=Path, required=True, help="Root directory containing repo feature dirs")
    parser.add_argument("--split-dir", type=Path, required=True, help="Directory with train.txt, val.txt, test.txt")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory for reports (default: inside checkpoint dir)")
    parser.add_argument("--eval-split", type=str, default="val", choices=["val", "test", "all"], help="Which split to evaluate on")
    parser.add_argument("--n-trials", type=int, default=5, help="Perturbation trials per graph")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device")
    parser.add_argument("--force", action="store_true", help="Re-run even if evaluation_report.json exists")
    args = parser.parse_args()

    device = torch.device(args.device)
    t0 = time.time()

    # Default output dir: save inside checkpoint directory
    output_dir = args.output_dir if args.output_dir else args.checkpoint

    # Skip if already evaluated
    existing_report = output_dir / "evaluation_report.json"
    if existing_report.exists() and not args.force:
        print(f"Evaluation already exists at {existing_report}")
        print("Use --force to re-run.")
        # Print existing report
        print("\n" + (output_dir / "evaluation_report.txt").read_text())
        return 0

    print(f"Reports will be saved to: {output_dir}/")

    # --- Load model ---
    print("\n[1/7] Loading model...")
    model = load_model(args.checkpoint, device)

    # Model info
    metadata_path = args.checkpoint / "metadata.json"
    model_info = {}
    if metadata_path.exists():
        with open(metadata_path) as f:
            model_info = json.load(f)
    model_info["checkpoint"] = str(args.checkpoint.resolve())
    model_info["n_params"] = sum(p.numel() for p in model.parameters())
    model_info["eval_split"] = args.eval_split
    model_info["n_trials"] = args.n_trials

    # --- Load dataset ---
    print("\n[2/7] Loading dataset...")
    if args.eval_split == "all":
        splits = ["val", "test"]
    else:
        splits = [args.eval_split]

    dataset = []
    for split in splits:
        split_file = args.split_dir / f"{split}.txt"
        if not split_file.exists():
            print(f"  Warning: {split_file} not found, skipping")
            continue
        print(f"  Loading {split} split:")
        dataset.extend(load_dataset(args.data_dir, split_file))

    if not dataset:
        print("ERROR: No graphs loaded. Check --data-dir and --split-dir paths.")
        sys.exit(1)

    print(f"  Total: {len(dataset)} graphs, {sum(d['node'].num_nodes for d in dataset)} nodes")

    # --- Tier 1: Intrinsic metrics ---
    print("\n[3/7] Running Tier 1: Intrinsic metrics...")
    t1 = time.time()
    tier1_results = tier1_intrinsic_metrics(model, dataset, device, n_trials=3)
    print(f"  Done in {time.time()-t1:.1f}s")
    print(f"  Recon sim: {tier1_results['recon_cosine_sim']:.4f}")
    print(f"  Cross-layer AUC: {tier1_results['crosslayer_auc']}")
    print(f"  R asymmetry: {tier1_results['R_asymmetry']:.4f}")

    # --- Tier 2: Phase 2 agreement ---
    print("\n[4/7] Running Tier 2: Phase 2 agreement...")
    t2 = time.time()
    tier2_results = tier2_phase2_agreement(model, dataset, device)
    print(f"  Done in {time.time()-t2:.1f}s")
    print(f"  Rank correlation: {tier2_results['rank_correlation_mean']:.4f}")
    print(f"  Top-k overlap: {tier2_results['topk_overlap_mean']:.4f}")

    # --- Tier 3: Synthetic perturbation ---
    print(f"\n[5/7] Running Tier 3: Synthetic perturbation ({args.n_trials} trials)...")
    t3 = time.time()
    tier3_results = tier3_perturbation_test(
        model, dataset, device, n_trials=args.n_trials,
    )
    print(f"  Done in {time.time()-t3:.1f}s")
    print(f"  Sensitivity: {tier3_results['perturbation_sensitivity_mean']:.4f}")
    print(f"  Specificity: {tier3_results['perturbation_specificity_mean']:.4f}")
    print(f"  Precision: {tier3_results['perturbation_precision_mean']:.4f}")
    print(f"  Control sens: {tier3_results['control_sensitivity_mean']:.4f}")

    # --- Tier 4: Structural consistency ---
    print("\n[6/7] Running Tier 4: Structural consistency...")
    t4 = time.time()
    tier4_results = tier4_structural_consistency(model, dataset, device)
    print(f"  Done in {time.time()-t4:.1f}s")
    print(f"  NMI: {tier4_results['nmi_mean']:.4f}")
    print(f"  Error-degree corr: {tier4_results['error_degree_corr_mean']:.4f}")

    # --- Baselines ---
    print(f"\n[7/7] Running ablation baselines ({args.n_trials} trials)...")
    t5 = time.time()
    baseline_results = run_baselines(dataset, n_trials=args.n_trials)
    print(f"  Done in {time.time()-t5:.1f}s")
    for name, res in baseline_results.items():
        print(f"  {name}: sens={res['sensitivity_mean']:.4f} prec={res['precision_mean']:.4f}")

    # --- Gate decision ---
    gate_result = go_no_go_gate(
        tier1_results, tier2_results, tier3_results, tier4_results, baseline_results,
    )

    # --- Generate report ---
    report = generate_report(
        tier1_results, tier2_results, tier3_results, tier4_results,
        baseline_results, gate_result,
        model_info=model_info,
        output_dir=output_dir,
    )

    print("\n" + report)
    total_time = time.time() - t0
    print(f"\nTotal evaluation time: {total_time:.1f}s")
    print(f"Reports saved to {output_dir}/")

    return 0 if gate_result["gate_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
