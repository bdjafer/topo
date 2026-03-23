"""Tier 2: Phase 2 Agreement (cross-method validation).

Phase 2's local_variation and Phase 3's reconstruction_error are independent
measures of the same thing: structural-semantic disagreement. If two methods
built on completely different principles agree, both are likely capturing
real signal.
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from torch_geometric.data import HeteroData

from topo_eval.phase2_proxy import compute_phase2_results


TARGETS = {
    "rank_correlation_mean": 0.3,
    "topk_overlap_mean": 0.3,
}


@torch.no_grad()
def _compute_reconstruction_error(model, data: HeteroData, device: torch.device) -> np.ndarray:
    """Per-node reconstruction error (cosine distance) with no masking.

    Args:
        model: Trained R-GIN.
        data: Single graph (unbatched).
        device: Torch device.

    Returns:
        [n_nodes] array of reconstruction errors.
    """
    from torch_geometric.loader import DataLoader

    loader = DataLoader([data], batch_size=1)
    batch = next(iter(loader)).to(device)

    mask = torch.zeros(batch["node"].num_nodes, dtype=torch.bool, device=device)
    z_str, *_ = model(batch, mask)
    pred = model.decode(z_str)
    target = batch["node"].x_semantic.float()

    # Per-node cosine distance
    errors = 1.0 - F.cosine_similarity(pred, target, dim=-1)
    return errors.cpu().numpy()


def tier2_phase2_agreement(
    model,
    dataset: list,
    device: torch.device,
    phase2_cache: dict | None = None,
) -> dict:
    """Cross-validate Phase 3 reconstruction error against Phase 2 local variation.

    Args:
        model: Trained R-GIN in eval mode.
        dataset: List of PyG HeteroData graphs.
        device: Torch device.
        phase2_cache: Optional precomputed Phase 2 results keyed by repo name.

    Returns:
        Dict with rank correlations, top-k overlaps, and pass/fail.
    """
    model.eval()

    rank_correlations = []
    topk_overlaps = []
    per_repo = {}

    for data in dataset:
        repo_key = getattr(data, "repo", "unknown")

        # Phase 3: reconstruction error per node
        p3_errors = _compute_reconstruction_error(model, data, device)

        # Phase 2: local variation per node
        if phase2_cache and repo_key in phase2_cache:
            p2_results = phase2_cache[repo_key]
        else:
            p2_results = compute_phase2_results(data)

        p2_scores = p2_results["local_variation"]

        n = len(p2_scores)
        if n < 5:
            continue

        # Spearman rank correlation
        rho, pval = stats.spearmanr(p2_scores, p3_errors)
        if np.isnan(rho):
            rho = 0.0
        rank_correlations.append(rho)

        # Top-k overlap (Jaccard of top 20% anomalous nodes)
        k = max(1, n // 5)
        top_p2 = set(np.argsort(p2_scores)[-k:])
        top_p3 = set(np.argsort(p3_errors)[-k:])
        jaccard = len(top_p2 & top_p3) / len(top_p2 | top_p3) if top_p2 | top_p3 else 0.0
        topk_overlaps.append(jaccard)

        per_repo[repo_key] = {
            "rank_correlation": float(rho),
            "topk_overlap": float(jaccard),
            "p_value": float(pval) if not np.isnan(pval) else None,
            "n_nodes": n,
        }

    results = {
        "rank_correlation_mean": float(np.mean(rank_correlations)) if rank_correlations else 0.0,
        "rank_correlation_std": float(np.std(rank_correlations)) if rank_correlations else 0.0,
        "topk_overlap_mean": float(np.mean(topk_overlaps)) if topk_overlaps else 0.0,
        "topk_overlap_std": float(np.std(topk_overlaps)) if topk_overlaps else 0.0,
        "per_repo": per_repo,
    }

    # Pass/fail
    results["passes"] = {}
    for key, target in TARGETS.items():
        val = results.get(key, 0.0)
        results["passes"][key] = val > target

    results["tier2_pass"] = all(results["passes"].values())

    return results
