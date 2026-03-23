"""Training objectives for R-GIN: 3 losses + HSIC regularizer."""

import torch
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Loss 1: Masked Semantic Feature Prediction
# ---------------------------------------------------------------------------

def masked_reconstruction_loss(predictions: Tensor, targets: Tensor) -> Tensor:
    """Cosine distance between predicted and target semantic embeddings.

    Args:
        predictions: [M, 768] decoded embeddings for masked nodes
        targets: [M, 768] raw CodeLM embeddings for masked nodes

    Returns:
        Scalar: mean cosine distance (1 - cos_sim)
    """
    if predictions.shape[0] == 0:
        return torch.tensor(0.0, device=predictions.device)
    return (1 - F.cosine_similarity(predictions, targets, dim=-1)).mean()


# ---------------------------------------------------------------------------
# Loss 2: Asymmetric Cross-Layer Edge Prediction
# ---------------------------------------------------------------------------

def cross_layer_score(z_imports: Tensor, edges: Tensor, R: Tensor) -> Tensor:
    """Compute bilinear edge scores: z_u^T R z_v.

    Args:
        z_imports: [N, 32] import-layer embeddings
        edges: [2, E] edge index (source, target)
        R: [32, 32] bilinear matrix

    Returns:
        [E] scores (logits, not probabilities)
    """
    z_u = z_imports[edges[0]]  # [E, 32]
    z_v = z_imports[edges[1]]  # [E, 32]
    # z_u^T R z_v = sum((z_u @ R) * z_v, dim=-1)
    return (z_u @ R * z_v).sum(dim=-1)


def cross_layer_loss(
    z_imports: Tensor,
    pos_edges: Tensor,
    neg_edges: Tensor,
    R: Tensor,
) -> Tensor:
    """BCE loss for cross-layer edge prediction.

    Args:
        z_imports: [N, 32] import-layer embeddings
        pos_edges: [2, P] positive call edges
        neg_edges: [2, Q] negative (random) edges
        R: [32, 32] bilinear matrix

    Returns:
        Scalar BCE loss
    """
    if pos_edges.shape[1] == 0:
        return torch.tensor(0.0, device=z_imports.device)

    pos_scores = cross_layer_score(z_imports, pos_edges, R)
    neg_scores = cross_layer_score(z_imports, neg_edges, R)

    scores = torch.cat([pos_scores, neg_scores])
    labels = torch.cat([
        torch.ones_like(pos_scores),
        torch.zeros_like(neg_scores),
    ])
    return F.binary_cross_entropy_with_logits(scores, labels)


def sample_negatives(batch, pos_edges: Tensor, ratio: int = 5) -> Tensor:
    """Sample negative edges for cross-layer prediction.

    For each positive call edge (u, v), sample `ratio` random non-call-neighbor
    nodes FROM THE SAME GRAPH (respecting batch boundaries).

    Args:
        batch: PyG batch with .batch (node→graph mapping) and .ptr (graph boundaries)
        pos_edges: [2, P] positive call edges
        ratio: negative-to-positive ratio

    Returns:
        [2, P*ratio] negative edges
    """
    if pos_edges.shape[1] == 0:
        return torch.zeros(2, 0, dtype=torch.long, device=pos_edges.device)

    device = pos_edges.device
    n_neg = pos_edges.shape[1] * ratio

    # Repeat each source node `ratio` times
    src = pos_edges[0].repeat(ratio)
    batch_idx = batch["node"].batch
    graph_ids = batch_idx[src]  # which graph each source belongs to

    # Get graph boundaries from ptr (HeteroData stores per node type)
    if hasattr(batch, "ptr") and batch.ptr is not None:
        ptr = batch.ptr
    else:
        ptr = batch["node"].ptr

    # Sample random targets per graph
    tgt = torch.zeros(n_neg, dtype=torch.long, device=device)
    for g_id in graph_ids.unique():
        gmask = graph_ids == g_id
        g_start = ptr[g_id].item()
        g_end = ptr[g_id + 1].item()
        tgt[gmask] = torch.randint(g_start, g_end, (gmask.sum(),), device=device)

    return torch.stack([src, tgt])


# ---------------------------------------------------------------------------
# Loss 3: Graph-Level Contrastive (InfoNCE)
# ---------------------------------------------------------------------------

def graph_contrastive_loss(g_emb: Tensor, pair_labels: Tensor, tau: float = 0.07) -> Tensor:
    """InfoNCE contrastive loss for graph-level embeddings.

    Expects pairs: g_emb[2i] and g_emb[2i+1] are positive pairs (same repo,
    different subgraph samples). All other pairs are negatives.

    Args:
        g_emb: [2B, 64] graph embeddings (pairs of subgraphs)
        pair_labels: unused, kept for interface compatibility
        tau: temperature

    Returns:
        Scalar InfoNCE loss
    """
    n = g_emb.shape[0]
    if n < 2:
        return torch.tensor(0.0, device=g_emb.device)

    # Normalize embeddings
    g_norm = F.normalize(g_emb, dim=-1)

    # Pairwise cosine similarity
    sim = g_norm @ g_norm.T / tau  # [2B, 2B]

    # Mask out self-similarity
    mask = torch.eye(n, dtype=torch.bool, device=g_emb.device)
    sim.masked_fill_(mask, float("-inf"))

    # For each anchor i, its positive is i^1 (XOR with 1: 0↔1, 2↔3, etc.)
    labels = torch.arange(n, device=g_emb.device)
    labels = labels ^ 1  # flip last bit

    return F.cross_entropy(sim, labels)


# ---------------------------------------------------------------------------
# HSIC Decorrelation Regularizer
# ---------------------------------------------------------------------------

def rbf_kernel(X: Tensor) -> Tensor:
    """RBF kernel with median bandwidth heuristic (Gretton 2012)."""
    dists = torch.cdist(X, X, p=2)  # [n, n]
    # Use upper triangle only (exclude diagonal zeros)
    n = dists.shape[0]
    upper_mask = torch.triu(torch.ones(n, n, dtype=torch.bool, device=X.device), diagonal=1)
    upper = dists[upper_mask]
    if upper.numel() == 0:
        return torch.ones(n, n, device=X.device)
    sigma = upper.median().clamp(min=1e-5)
    return torch.exp(-dists.pow(2) / (2 * sigma.pow(2)))


def hsic(X: Tensor, Y: Tensor) -> Tensor:
    """Biased HSIC estimator.

    Args:
        X: [n, d1]
        Y: [n, d2]

    Returns:
        Scalar HSIC value (clamped to non-negative)
    """
    n = X.shape[0]
    if n < 5:
        return torch.tensor(0.0, device=X.device)

    K_X = rbf_kernel(X)
    K_Y = rbf_kernel(Y)

    # Centering matrix H = I - (1/n) 11^T
    # HSIC = (1/n²) tr(K_X H K_Y H)
    H = torch.eye(n, device=X.device) - 1.0 / n
    HK_X = H @ K_X
    HK_Y = H @ K_Y
    return ((HK_X * HK_Y.T).sum() / (n * n)).clamp(min=0)


def per_graph_hsic(
    batch_idx: Tensor,
    z_inv: Tensor,
    z_calls: Tensor,
    z_imports: Tensor,
    z_inherits: Tensor,
    has_inherits: bool = True,
) -> Tensor:
    """Compute HSIC decorrelation regularizer, averaged per graph.

    Args:
        batch_idx: [N] graph assignment for each node
        z_inv: [N, 64] invariant embeddings
        z_calls: [N, 32] calls embeddings
        z_imports: [N, 32] imports embeddings
        z_inherits: [N, 32] inherits embeddings
        has_inherits: whether to include inherits term

    Returns:
        Scalar: mean HSIC across graphs and pairs
    """
    device = z_inv.device
    total_hsic = torch.tensor(0.0, device=device)
    n_graphs = 0

    for g_id in batch_idx.unique():
        gmask = batch_idx == g_id
        zi = z_inv[gmask]
        zc = z_calls[gmask]
        zm = z_imports[gmask]

        graph_hsic = hsic(zi, zc) + hsic(zi, zm)
        if has_inherits:
            zh = z_inherits[gmask]
            graph_hsic = graph_hsic + hsic(zi, zh)

        total_hsic = total_hsic + graph_hsic
        n_graphs += 1

    if n_graphs == 0:
        return torch.tensor(0.0, device=device)
    return total_hsic / n_graphs


# ---------------------------------------------------------------------------
# Mask Generation
# ---------------------------------------------------------------------------

def generate_mask(batch, ratio: float = 0.65) -> Tensor:
    """Generate a boolean mask (True = masked) per node, respecting per-graph boundaries.

    Args:
        batch: PyG batch with .ptr (graph boundary pointers)
        ratio: fraction of nodes to mask per graph

    Returns:
        [N] boolean tensor
    """
    # HeteroData batches store ptr per node type
    if hasattr(batch, "ptr") and batch.ptr is not None:
        ptr = batch.ptr
    else:
        ptr = batch["node"].ptr
    device = ptr.device
    masks = []
    for i in range(len(ptr) - 1):
        n = (ptr[i + 1] - ptr[i]).item()
        n_mask = max(1, int(n * ratio))
        perm = torch.randperm(n, device=device)
        graph_mask = torch.zeros(n, dtype=torch.bool, device=device)
        graph_mask[perm[:n_mask]] = True
        masks.append(graph_mask)
    return torch.cat(masks)


# ---------------------------------------------------------------------------
# Loss Weight Ramp
# ---------------------------------------------------------------------------

def ramp(epoch: int, start: int, end: int, target: float) -> float:
    """Linear ramp from 0 to target between start and end epochs."""
    if epoch < start:
        return 0.0
    if epoch >= end:
        return target
    return target * (epoch - start) / (end - start)
