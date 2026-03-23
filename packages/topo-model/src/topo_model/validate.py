"""Validation and intrinsic metrics for R-GIN training."""

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from sklearn.metrics import roc_auc_score

from topo_model.losses import (
    cross_layer_score,
    generate_mask,
    sample_negatives,
)


@torch.no_grad()
def validate(model, val_loader, device: torch.device, mask_ratio: float = 0.65) -> dict:
    """Compute validation metrics.

    Metrics computed:
    - recon_cosine_sim: masked reconstruction cosine similarity (primary)
    - crosslayer_auc: import→call edge prediction AUC
    - R_asymmetry: asymmetry ratio of bilinear matrix R

    Args:
        model: R-GIN model in eval mode
        val_loader: PyG DataLoader for validation set
        device: torch device
        mask_ratio: masking ratio for reconstruction

    Returns:
        Dict of metric name → value
    """
    model.eval()
    recon_sims = []
    crosslayer_preds = []
    crosslayer_labels = []

    for batch in val_loader:
        batch = batch.to(device)
        mask = generate_mask(batch, ratio=mask_ratio)

        z_str, z_inv, z_calls, z_imports, z_inherits, g_emb = model(batch, mask)

        # --- Reconstruction similarity (on masked nodes only) ---
        if mask.any():
            pred = model.decode(z_str[mask])
            target = batch["node"].x_semantic[mask].float()
            sim = F.cosine_similarity(pred, target, dim=-1).mean()
            recon_sims.append(sim.item())

        # --- Cross-layer AUC ---
        call_key = ("node", "calls", "node")
        if call_key in batch.edge_types and batch[call_key].edge_index.shape[1] > 0:
            pos_edges = batch[call_key].edge_index
            neg_edges = sample_negatives(batch, pos_edges, ratio=5)
            pos_scores = cross_layer_score(z_imports, pos_edges, model.R)
            neg_scores = cross_layer_score(z_imports, neg_edges, model.R)
            crosslayer_preds.extend(pos_scores.cpu().tolist())
            crosslayer_preds.extend(neg_scores.cpu().tolist())
            crosslayer_labels.extend([1] * len(pos_scores))
            crosslayer_labels.extend([0] * len(neg_scores))

    # --- Compute aggregate metrics ---
    metrics = {}

    if recon_sims:
        metrics["recon_cosine_sim"] = float(np.mean(recon_sims))
    else:
        metrics["recon_cosine_sim"] = 0.0

    if crosslayer_labels:
        try:
            metrics["crosslayer_auc"] = float(
                roc_auc_score(crosslayer_labels, crosslayer_preds)
            )
        except ValueError:
            metrics["crosslayer_auc"] = 0.5  # degenerate case
    else:
        metrics["crosslayer_auc"] = None

    # R asymmetry ratio
    R = model.R.detach().cpu().numpy()
    r_norm = np.linalg.norm(R)
    if r_norm > 1e-8:
        metrics["R_asymmetry"] = float(np.linalg.norm(R - R.T) / r_norm)
    else:
        metrics["R_asymmetry"] = 0.0

    model.train()
    return metrics
