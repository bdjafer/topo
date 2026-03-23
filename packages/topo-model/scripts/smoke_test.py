#!/usr/bin/env python3
"""Smoke test: run 5 epochs on available data, verify no NaN/crash.

Usage:
    python scripts/smoke_test.py
    python scripts/smoke_test.py --epochs 10
"""

import argparse
import sys
from pathlib import Path

import torch
import numpy as np

# Add topo-dataset to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "packages" / "topo-dataset" / "src"))

from topo_dataset.loader import load_graph
from torch_geometric.loader import DataLoader

from topo_model.config import RGINConfig
from topo_model.rgin import RGIN
from topo_model.losses import (
    masked_reconstruction_loss,
    cross_layer_loss,
    per_graph_hsic,
    generate_mask,
    sample_negatives,
    ramp,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    examples_dir = PROJECT_ROOT / "examples"

    # Find all repos with features.npz
    repos = sorted(
        d.name for d in examples_dir.iterdir()
        if d.is_dir() and (d / "features.npz").exists()
    )

    if not repos:
        print("ERROR: No preprocessed repos found. Run preprocess.py first.")
        sys.exit(1)

    print(f"Found {len(repos)} preprocessed repos: {repos}")

    # Load graphs
    graphs = []
    for name in repos:
        try:
            g = load_graph(examples_dir / name)
            graphs.append(g)
            n = g["node"].num_nodes
            print(f"  {name}: {n} nodes")
        except Exception as e:
            print(f"  {name}: SKIP ({e})")

    if not graphs:
        print("ERROR: No valid graphs loaded.")
        sys.exit(1)

    # Use smaller config for smoke test
    config = RGINConfig(
        hidden_dim=64,
        n_layers=2,
        epochs=args.epochs,
        batch_size=min(len(graphs), 4),
        lr=1e-3,
        eval_every=1,
        save_every=999,
        warmup_epochs=2,
        ramp_start=2,
        ramp_end=4,
        decay_start=4,
    )

    device = torch.device(args.device)
    model = RGIN(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {n_params:,} params (test config, hidden=64)")

    loader = DataLoader(graphs, batch_size=config.batch_size, shuffle=True)

    print(f"\nRunning {args.epochs} epochs...")
    for epoch in range(args.epochs):
        model.train()
        epoch_losses = []

        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            mask = generate_mask(batch, ratio=0.65)
            z_str, z_inv, z_calls, z_imports, z_inherits, g_emb = model(batch, mask)

            # Loss 1: reconstruction
            pred = model.decode(z_str[mask])
            target = batch["node"].x_semantic[mask].float()
            loss_recon = masked_reconstruction_loss(pred, target)

            # Loss 2: cross-layer
            call_key = ("node", "calls", "node")
            if call_key in batch.edge_types and batch[call_key].edge_index.shape[1] > 0:
                pos_edges = batch[call_key].edge_index
                neg_edges = sample_negatives(batch, pos_edges, ratio=5)
                loss_cross = cross_layer_loss(z_imports, pos_edges, neg_edges, model.R)
            else:
                loss_cross = torch.tensor(0.0, device=device)

            # HSIC
            batch_idx = batch["node"].batch
            loss_decorr = per_graph_hsic(batch_idx, z_inv, z_calls, z_imports, z_inherits)

            # Combined
            alpha = ramp(epoch, config.ramp_start, config.ramp_end, 0.5)
            lam = ramp(epoch, config.ramp_start, config.ramp_end, 0.01)
            loss = loss_recon + alpha * loss_cross + lam * loss_decorr

            # Check for NaN/Inf
            if not torch.isfinite(loss):
                print(f"ERROR: Non-finite loss at epoch {epoch}!")
                print(f"  recon={loss_recon.item()}, cross={loss_cross.item()}, decorr={loss_decorr.item()}")
                sys.exit(1)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_losses.append(loss.item())

        avg_loss = sum(epoch_losses) / len(epoch_losses)

        # Quick validation
        model.eval()
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                mask = generate_mask(batch, ratio=0.65)
                z_str, *_ = model(batch, mask)
                pred = model.decode(z_str[mask])
                target = batch["node"].x_semantic[mask].float()
                val_sim = torch.nn.functional.cosine_similarity(pred, target, dim=-1).mean().item()
                break  # just first batch

        print(f"  Epoch {epoch}: loss={avg_loss:.4f}, val_sim={val_sim:.4f}")

    print("\nSmoke test PASSED — no NaN/Inf, training loop functional.")

    # Final sanity: check all outputs are finite
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            mask = torch.zeros(batch["node"].x_semantic.shape[0], dtype=torch.bool)
            outputs = model(batch, mask)
            for i, out in enumerate(outputs):
                assert torch.isfinite(out).all(), f"Non-finite output {i}"
            break

    print("All outputs finite. Model is healthy.")


if __name__ == "__main__":
    main()
