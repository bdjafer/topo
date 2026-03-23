#!/usr/bin/env python3
"""Full training run for R-GIN model on available data.

Loads preprocessed repos, trains with full architecture,
evaluates, and exports model bundle.

Usage:
    python scripts/run_training.py
    python scripts/run_training.py --epochs 100 --device mps
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Add topo-dataset to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "packages" / "topo-dataset" / "src"))

from topo_dataset.loader import load_graph
from torch_geometric.loader import DataLoader

from topo_model.config import RGINConfig
from topo_model.rgin import RGIN
from topo_model.train import train_epoch, get_lr
from topo_model.validate import validate
from topo_model.losses import generate_mask, ramp
from topo_model.export import export_bundle


def load_split(examples_dir: Path, split_file: Path) -> list:
    """Load graphs for a split."""
    repos = [l.strip() for l in split_file.read_text().strip().split("\n") if l.strip()]
    graphs = []
    for name in repos:
        repo_dir = examples_dir / name
        if (repo_dir / "features.npz").exists():
            try:
                g = load_graph(repo_dir)
                graphs.append(g)
                print(f"  {name}: {g['node'].num_nodes} nodes")
            except Exception as e:
                print(f"  {name}: SKIP ({e})")
        else:
            print(f"  {name}: no features.npz")
    return graphs


def main():
    parser = argparse.ArgumentParser(description="Train R-GIN model")
    parser.add_argument("--epochs", type=int, default=100, help="Max epochs")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device: cpu, mps, cuda")
    parser.add_argument("--hidden-dim", type=int, default=256,
                        help="Hidden dimension (256=full, 128=medium)")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size (graphs per batch)")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output directory")
    args = parser.parse_args()

    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)
    print(f"Device: {device}")

    examples_dir = PROJECT_ROOT / "examples"
    splits_dir = examples_dir / "splits"

    # Load data
    print("\n=== Loading training data ===")
    train_graphs = load_split(examples_dir, splits_dir / "train.txt")
    print(f"\n=== Loading validation data ===")
    val_graphs = load_split(examples_dir, splits_dir / "val.txt")

    if not train_graphs:
        print("ERROR: No training graphs loaded.")
        sys.exit(1)

    total_train_nodes = sum(g["node"].num_nodes for g in train_graphs)
    total_val_nodes = sum(g["node"].num_nodes for g in val_graphs)
    print(f"\nTrain: {len(train_graphs)} graphs, {total_train_nodes:,} nodes")
    print(f"Val:   {len(val_graphs)} graphs, {total_val_nodes:,} nodes")

    # Config
    config = RGINConfig(
        hidden_dim=args.hidden_dim,
        n_layers=2,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        eval_every=5,
        save_every=25,
        warmup_epochs=5,
        ramp_start=5,
        ramp_end=20,
        decay_start=20,
        patience=30,
    )

    # Output dir
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = PROJECT_ROOT / "checkpoints" / f"rgin_h{args.hidden_dim}_e{args.epochs}"
    out_dir.mkdir(parents=True, exist_ok=True)
    config.save(out_dir / "config.json")

    # Build model
    model = RGIN(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {n_params:,} params ({n_params/1e6:.2f}M)")
    print(f"Data/param ratio: {total_train_nodes / n_params:.1f}:1")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    # Data loaders
    train_loader = DataLoader(train_graphs, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=config.batch_size, shuffle=False)

    # Training loop
    best_val_sim = -1.0
    patience_counter = 0
    history = []
    start_time = time.time()

    print(f"\n{'='*60}")
    print(f"Starting training: {config.epochs} epochs, lr={config.lr}")
    print(f"{'='*60}\n")

    for epoch in range(config.epochs):
        epoch_start = time.time()

        # Update LR
        lr = get_lr(epoch, config)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Train epoch
        train_metrics = train_epoch(model, train_loader, optimizer, epoch, config, device)

        # Validate
        val_metrics = {}
        if epoch % config.eval_every == 0 or epoch == config.epochs - 1:
            val_metrics = validate(model, val_loader, device, config.mask_ratio)

        epoch_time = time.time() - epoch_start

        # Log
        record = {
            "epoch": epoch,
            "lr": lr,
            "epoch_time": epoch_time,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(record)

        # Print
        val_sim = val_metrics.get("recon_cosine_sim")
        val_auc = val_metrics.get("crosslayer_auc")
        val_str = ""
        if val_sim is not None:
            val_str = f" | val_sim={val_sim:.4f}"
        if val_auc is not None:
            val_str += f" auc={val_auc:.4f}"

        print(
            f"Epoch {epoch:3d}/{config.epochs} | "
            f"lr={lr:.1e} | "
            f"loss={train_metrics['loss']:.4f} "
            f"(R={train_metrics['recon']:.4f} "
            f"C={train_metrics['cross']:.4f} "
            f"D={train_metrics['decorr']:.4f})"
            f"{val_str} | {epoch_time:.1f}s"
        )

        # Best model tracking
        if val_sim is not None:
            if val_sim > best_val_sim:
                best_val_sim = val_sim
                patience_counter = 0
                torch.save({
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_metric": val_sim,
                    "config": config.__dict__,
                }, out_dir / "best_model.pt")
                print(f"  -> New best: {val_sim:.4f}")
            else:
                patience_counter += config.eval_every

        # Periodic checkpoint
        if epoch % config.save_every == 0:
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_metric": val_sim if val_sim else -1,
                "config": config.__dict__,
            }, out_dir / f"checkpoint_{epoch:04d}.pt")

        # Early stopping
        if patience_counter >= config.patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    total_time = time.time() - start_time

    # Final evaluation
    print(f"\n{'='*60}")
    print(f"Training complete in {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"{'='*60}")

    final_metrics = validate(model, val_loader, device, config.mask_ratio)
    print(f"\nFinal metrics:")
    for k, v in final_metrics.items():
        if v is not None:
            print(f"  {k}: {v:.4f}")

    print(f"\nBest val_sim: {best_val_sim:.4f}")

    # Save final model
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "val_metric": best_val_sim,
        "config": config.__dict__,
    }, out_dir / "final_model.pt")

    # Save history
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2, default=str)

    # Save metadata
    metadata = {
        "n_params": n_params,
        "n_train_graphs": len(train_graphs),
        "n_val_graphs": len(val_graphs),
        "n_train_nodes": total_train_nodes,
        "n_val_nodes": total_val_nodes,
        "epochs_trained": epoch + 1,
        "best_val_sim": best_val_sim,
        "final_metrics": final_metrics,
        "total_time_s": total_time,
        "device": str(device),
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Export model bundle
    print(f"\nExporting model bundle to {out_dir}/")
    export_bundle(out_dir / "best_model.pt", out_dir)

    # --- Evidence of learning ---
    print(f"\n{'='*60}")
    print("EVIDENCE OF LEARNING")
    print(f"{'='*60}")

    # Check if loss decreased
    first_loss = history[0]["train_loss"]
    last_loss = history[-1]["train_loss"]
    print(f"  Train loss: {first_loss:.4f} → {last_loss:.4f} (Δ={first_loss - last_loss:.4f})")

    # Check val_sim improved
    val_sims = [r.get("val_recon_cosine_sim") for r in history if r.get("val_recon_cosine_sim") is not None]
    if len(val_sims) >= 2:
        print(f"  Val cosine sim: {val_sims[0]:.4f} → {val_sims[-1]:.4f} (Δ={val_sims[-1] - val_sims[0]:.4f})")
        if val_sims[-1] > val_sims[0]:
            print("  ✓ Model is learning — val cosine sim improved")
        else:
            print("  ✗ Warning: val cosine sim did not improve")

    # Check cross-layer AUC
    aucs = [r.get("val_crosslayer_auc") for r in history if r.get("val_crosslayer_auc") is not None]
    if len(aucs) >= 2:
        print(f"  Cross-layer AUC: {aucs[0]:.4f} → {aucs[-1]:.4f}")
        if aucs[-1] > 0.6:
            print("  ✓ Cross-layer prediction learned")

    # R asymmetry
    if final_metrics.get("R_asymmetry") is not None:
        asym = final_metrics["R_asymmetry"]
        print(f"  R asymmetry: {asym:.4f} ({'directional ✓' if asym >= 0.1 else 'symmetric — fallback to binary'})")

    print(f"\nModel bundle saved to: {out_dir}")


if __name__ == "__main__":
    main()
