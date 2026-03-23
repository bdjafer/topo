"""Training loop for R-GIN model."""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from topo_model.config import RGINConfig
from topo_model.rgin import RGIN
from topo_model.losses import (
    masked_reconstruction_loss,
    cross_layer_loss,
    per_graph_hsic,
    generate_mask,
    ramp,
    sample_negatives,
)
from topo_model.validate import validate


def get_lr(epoch: int, config: RGINConfig) -> float:
    """Compute learning rate for given epoch.

    Schedule:
    - Epochs 0..warmup: linear warmup 0 → lr
    - Epochs warmup..decay_start: constant lr
    - Epochs decay_start..end: cosine decay lr → lr_min
    """
    if epoch < config.warmup_epochs:
        # Linear warmup
        return config.lr * (epoch + 1) / config.warmup_epochs
    elif epoch < config.decay_start:
        # Constant
        return config.lr
    else:
        # Cosine decay
        progress = (epoch - config.decay_start) / max(1, config.epochs - config.decay_start)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return config.lr_min + (config.lr - config.lr_min) * cosine


def train_epoch(
    model: RGIN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: RGINConfig,
    device: torch.device,
) -> dict:
    """Run one training epoch.

    Returns dict with per-loss averages for logging.
    """
    model.train()

    # Compute loss weight ramps
    alpha = ramp(epoch, config.ramp_start, config.ramp_end, config.alpha_crosslayer)
    beta = ramp(epoch, config.ramp_start, config.ramp_end, config.beta_graph)
    lam = ramp(epoch, config.ramp_start, config.ramp_end, config.lambda_decorr)

    total_loss = 0.0
    total_recon = 0.0
    total_cross = 0.0
    total_graph = 0.0
    total_decorr = 0.0
    n_batches = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        # 1. Generate mask
        mask = generate_mask(batch, ratio=config.mask_ratio)

        # 2. Forward pass
        z_str, z_inv, z_calls, z_imports, z_inherits, _g_emb = model(batch, mask)

        # 3. Loss 1: Masked reconstruction
        predictions = model.decode(z_str[mask])
        targets = batch["node"].x_semantic[mask].float()
        loss_recon = masked_reconstruction_loss(predictions, targets)

        # 4. Loss 2: Cross-layer edge prediction
        call_key = ("node", "calls", "node")
        if call_key in batch.edge_types and batch[call_key].edge_index.shape[1] > 0:
            pos_edges = batch[call_key].edge_index
            neg_edges = sample_negatives(batch, pos_edges, ratio=config.neg_ratio)
            loss_cross = cross_layer_loss(z_imports, pos_edges, neg_edges, model.R)
        else:
            loss_cross = torch.tensor(0.0, device=device)

        # 5. Loss 3: Graph contrastive
        # DISABLED without paired subgraph sampling — using unpaired graphs
        # as "positives" would push different repos together (harmful).
        # Enable via use_contrastive_loader=True which provides proper pairs.
        loss_graph = torch.tensor(0.0, device=device)

        # 6. HSIC decorrelation
        batch_idx = batch["node"].batch
        inherits_key = ("node", "inherits", "node")
        has_inherits = (
            inherits_key in batch.edge_types
            and batch[inherits_key].edge_index.shape[1] > 0
        )
        loss_decorr = per_graph_hsic(batch_idx, z_inv, z_calls, z_imports, z_inherits, has_inherits)

        # 7. Combined loss
        loss = loss_recon + alpha * loss_cross + beta * loss_graph + lam * loss_decorr

        # 8. Backward + optimize
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip)
        optimizer.step()

        total_loss += loss.item()
        total_recon += loss_recon.item()
        total_cross += loss_cross.item()
        total_graph += loss_graph.item()
        total_decorr += loss_decorr.item()
        n_batches += 1

    n = max(n_batches, 1)
    return {
        "loss": total_loss / n,
        "recon": total_recon / n,
        "cross": total_cross / n,
        "graph": total_graph / n,
        "decorr": total_decorr / n,
        "alpha": alpha,
        "beta": beta,
        "lambda": lam,
    }


def build_contrastive_loader(dataset, batch_size: int, seed: int = 42):
    """Build a DataLoader that yields pairs of subgraph samples per repo.

    For graph contrastive loss, each repo produces 2 subgraph views.
    The loader yields batches of 2*batch_size graphs (paired).
    """
    from topo_dataset.transforms import sample_subgraph

    class PairedDataset:
        def __init__(self, base_dataset, seed):
            self.base = base_dataset
            self.seed = seed

        def __len__(self):
            return len(self.base) * 2

        def __getitem__(self, idx):
            repo_idx = idx // 2
            view = idx % 2
            data = self.base[repo_idx]
            # Each view uses a different seed for different subgraph
            return sample_subgraph(data, ratio=0.7, seed=self.seed + idx)

    paired = PairedDataset(dataset, seed)
    return DataLoader(paired, batch_size=batch_size * 2, shuffle=True)


def train(
    train_dataset,
    val_dataset,
    config: RGINConfig,
    checkpoint_dir: Path,
    device: torch.device,
    use_contrastive_loader: bool = False,
) -> dict:
    """Full training loop.

    Args:
        train_dataset: PyG Dataset for training
        val_dataset: PyG Dataset for validation
        config: model configuration
        checkpoint_dir: where to save checkpoints
        device: torch device
        use_contrastive_loader: if True, use paired subgraph loader for contrastive loss

    Returns:
        Dict with training results
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config.save(checkpoint_dir / "config.json")

    # Build model
    model = RGIN(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,} ({n_params/1e6:.2f}M)")

    # Build optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    # Build data loaders
    if use_contrastive_loader and len(train_dataset) >= 2:
        train_loader = build_contrastive_loader(train_dataset, config.batch_size)
    else:
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    # Training state
    best_val_sim = -1.0
    patience_counter = 0
    history = []

    start_time = time.time()

    for epoch in range(config.epochs):
        epoch_start = time.time()

        # Update learning rate
        lr = get_lr(epoch, config)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, epoch, config, device)

        # Validate
        if epoch % config.eval_every == 0 or epoch == config.epochs - 1:
            val_metrics = validate(model, val_loader, device, config.mask_ratio)
        else:
            val_metrics = {}

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

        # Print progress
        val_sim = val_metrics.get("recon_cosine_sim", None)
        val_auc = val_metrics.get("crosslayer_auc", None)
        val_str = ""
        if val_sim is not None:
            val_str = f" | val_sim={val_sim:.4f}"
        if val_auc is not None:
            val_str += f" auc={val_auc:.4f}"

        print(
            f"Epoch {epoch:3d}/{config.epochs} | "
            f"lr={lr:.2e} | "
            f"loss={train_metrics['loss']:.4f} "
            f"(recon={train_metrics['recon']:.4f} "
            f"cross={train_metrics['cross']:.4f} "
            f"graph={train_metrics['graph']:.4f} "
            f"decorr={train_metrics['decorr']:.4f})"
            f"{val_str} | {epoch_time:.1f}s"
        )

        # Checkpointing
        if val_sim is not None:
            if val_sim > best_val_sim:
                best_val_sim = val_sim
                patience_counter = 0
                _save_checkpoint(model, optimizer, epoch, val_sim, config, checkpoint_dir / "best_model.pt")
                print(f"  -> New best: val_sim={val_sim:.4f}")
            else:
                patience_counter += config.eval_every

        if epoch % config.save_every == 0:
            _save_checkpoint(
                model, optimizer, epoch,
                val_sim if val_sim is not None else -1,
                config, checkpoint_dir / f"checkpoint_epoch_{epoch:04d}.pt",
            )

        # Early stopping
        if patience_counter >= config.patience:
            print(f"Early stopping at epoch {epoch} (patience={config.patience})")
            break

    total_time = time.time() - start_time

    # Final validation
    final_metrics = validate(model, val_loader, device, config.mask_ratio)
    print(f"\nTraining complete in {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"Best val_sim: {best_val_sim:.4f}")
    print(f"Final metrics: {final_metrics}")

    # Save final checkpoint
    _save_checkpoint(model, optimizer, epoch, best_val_sim, config, checkpoint_dir / "final_model.pt")

    # Save training history
    with open(checkpoint_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2, default=str)

    # Save metadata
    metadata = {
        "n_params": n_params,
        "n_train": len(train_dataset),
        "n_val": len(val_dataset),
        "epochs_trained": epoch + 1,
        "best_val_sim": best_val_sim,
        "final_metrics": final_metrics,
        "total_time_s": total_time,
        "device": str(device),
    }
    with open(checkpoint_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return {
        "model": model,
        "history": history,
        "best_val_sim": best_val_sim,
        "final_metrics": final_metrics,
        "total_time": total_time,
    }


def _save_checkpoint(model, optimizer, epoch, val_metric, config, path):
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "val_metric": val_metric,
        "config": config.__dict__ if hasattr(config, "__dict__") else {},
    }, path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train R-GIN model")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to examples/ directory")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Output directory")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--device", type=str, default="auto", help="cpu, cuda, mps, or auto")
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Load datasets
    data_dir = Path(args.data_dir)
    splits_dir = data_dir / "splits"

    if not splits_dir.exists():
        print("ERROR: No splits directory found. Run split.py first.", file=sys.stderr)
        sys.exit(1)

    # Import topo_dataset
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "topo-dataset" / "src"))
    from topo_dataset import TopoDataset

    train_dataset = TopoDataset(data_dir, splits_dir / "train.txt")
    val_dataset = TopoDataset(data_dir, splits_dir / "val.txt")

    print(f"Train: {len(train_dataset)} graphs, Val: {len(val_dataset)} graphs")

    # Config
    config = RGINConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        save_every=args.save_every,
        eval_every=args.eval_every,
    )

    # Train
    results = train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config,
        checkpoint_dir=Path(args.checkpoint_dir),
        device=device,
    )

    print(f"\nDone. Best val_sim: {results['best_val_sim']:.4f}")


if __name__ == "__main__":
    main()
