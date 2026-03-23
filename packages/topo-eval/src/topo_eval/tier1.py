"""Tier 1: Intrinsic metrics from training objectives.

These are necessary but not sufficient — a model can overfit to reconstruction
without capturing meaningful structural patterns.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score

from topo_model.losses import cross_layer_score, generate_mask, sample_negatives


TARGETS = {
    "recon_cosine_sim": 0.6,
    "crosslayer_auc": 0.75,
    "R_asymmetry": 0.1,
}


@torch.no_grad()
def tier1_intrinsic_metrics(
    model,
    dataset: list,
    device: torch.device,
    mask_ratio: float = 0.65,
    n_trials: int = 3,
) -> dict:
    """Compute Tier 1 intrinsic metrics on a dataset.

    Runs multiple masking trials to reduce variance from random masks.

    Args:
        model: Trained R-GIN model.
        dataset: List of PyG HeteroData graphs.
        device: Torch device.
        mask_ratio: Fraction of nodes to mask.
        n_trials: Number of masking trials per graph (averaged).

    Returns:
        Dict with metrics and pass/fail per target.
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    per_graph_sims = []
    all_crosslayer_preds = []
    all_crosslayer_labels = []

    for batch in loader:
        batch = batch.to(device)
        trial_sims = []

        for _ in range(n_trials):
            mask = generate_mask(batch, ratio=mask_ratio)
            z_str, z_inv, z_calls, z_imports, z_inherits, g_emb = model(batch, mask)

            # Reconstruction similarity on masked nodes
            if mask.any():
                pred = model.decode(z_str[mask])
                target = batch["node"].x_semantic[mask].float()
                sim = F.cosine_similarity(pred, target, dim=-1).mean().item()
                trial_sims.append(sim)

        if trial_sims:
            per_graph_sims.append(np.mean(trial_sims))

        # Cross-layer AUC (no masking for this — we want full graph)
        mask_none = torch.zeros(batch["node"].num_nodes, dtype=torch.bool, device=device)
        z_str, z_inv, z_calls, z_imports, z_inherits, g_emb = model(batch, mask_none)

        call_key = ("node", "calls", "node")
        if call_key in batch.edge_types and batch[call_key].edge_index.shape[1] > 0:
            pos_edges = batch[call_key].edge_index
            neg_edges = sample_negatives(batch, pos_edges, ratio=5)
            pos_scores = cross_layer_score(z_imports, pos_edges, model.R)
            neg_scores = cross_layer_score(z_imports, neg_edges, model.R)
            all_crosslayer_preds.extend(pos_scores.cpu().tolist())
            all_crosslayer_preds.extend(neg_scores.cpu().tolist())
            all_crosslayer_labels.extend([1] * len(pos_scores))
            all_crosslayer_labels.extend([0] * len(neg_scores))

    # Aggregate
    metrics = {}
    metrics["recon_cosine_sim"] = float(np.mean(per_graph_sims)) if per_graph_sims else 0.0
    metrics["recon_cosine_sim_std"] = float(np.std(per_graph_sims)) if per_graph_sims else 0.0
    metrics["recon_cosine_sim_per_graph"] = per_graph_sims

    if all_crosslayer_labels:
        try:
            metrics["crosslayer_auc"] = float(
                roc_auc_score(all_crosslayer_labels, all_crosslayer_preds)
            )
        except ValueError:
            metrics["crosslayer_auc"] = 0.5
    else:
        metrics["crosslayer_auc"] = None

    R = model.R.detach().cpu().numpy()
    r_norm = np.linalg.norm(R)
    metrics["R_asymmetry"] = float(np.linalg.norm(R - R.T) / r_norm) if r_norm > 1e-8 else 0.0

    # Pass/fail
    metrics["passes"] = {}
    for key, target in TARGETS.items():
        val = metrics.get(key)
        if val is None:
            metrics["passes"][key] = None
        else:
            metrics["passes"][key] = val > target

    metrics["tier1_pass"] = all(
        v is True for v in metrics["passes"].values() if v is not None
    )

    return metrics
