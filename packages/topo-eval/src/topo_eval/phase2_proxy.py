"""Phase 2 proxy metrics computed in Python.

Replicates the key Phase 2 metrics from the Rust analyzer (semantic.rs)
so we can cross-validate Phase 3 without requiring the Rust binary.

These are computed directly from the features.npz data (graph structure +
semantic embeddings).
"""

import numpy as np
from torch_geometric.data import HeteroData


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    dot = np.dot(a, b)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(dot / (na * nb))


def compute_local_variation(data: HeteroData) -> np.ndarray:
    """Per-node local variation: avg semantic disagreement with neighbors.

    Mirrors semantic.rs::compute_local_variation:
        variation(n) = (1/deg) * sum_j w_nj * (1 - cos_sim(M[n], M[j]))

    Uses symmetric neighbors across all edge types (calls, imports, inherits).

    Args:
        data: PyG HeteroData with x_semantic and edge indices.

    Returns:
        [n_nodes] array of local variation scores.
    """
    n = data["node"].num_nodes
    semantic = data["node"].x_semantic.numpy()
    # Precompute norms for fast cosine sim
    norms = np.linalg.norm(semantic, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)
    normed = semantic / norms

    # Collect symmetric neighbors from all edge types
    neighbors: dict[int, set[int]] = {i: set() for i in range(n)}
    for edge_type in ["calls", "imports", "inherits"]:
        key = ("node", edge_type, "node")
        if key in data.edge_types:
            ei = data[key].edge_index.numpy()
            for j in range(ei.shape[1]):
                src, tgt = int(ei[0, j]), int(ei[1, j])
                neighbors[src].add(tgt)
                neighbors[tgt].add(src)

    variation = np.zeros(n, dtype=np.float64)
    for i in range(n):
        nbrs = neighbors[i]
        if not nbrs:
            variation[i] = 0.0
            continue
        nbr_list = list(nbrs)
        # Vectorized cosine sim: dot products of normed vectors
        sims = normed[nbr_list] @ normed[i]
        variation[i] = float(np.mean(1.0 - sims))

    return variation


def compute_module_assignments(data: HeteroData, n_clusters: int | None = None) -> np.ndarray:
    """Assign nodes to modules via spectral clustering on the combined adjacency.

    Uses the spectral eigenvectors already in the features for k-means clustering.

    Args:
        data: PyG HeteroData.
        n_clusters: Number of clusters. If None, uses sqrt(n_nodes) clamped to [2, 20].

    Returns:
        [n_nodes] array of module IDs.
    """
    from sklearn.cluster import KMeans

    n = data["node"].num_nodes
    if n_clusters is None:
        n_clusters = max(2, min(20, int(np.sqrt(n))))

    # Use spectral eigenvectors for clustering
    vecs = data["node"].x_spectral_vecs.numpy()
    k = min(n_clusters, vecs.shape[1])
    X = vecs[:, :k]

    # Normalize rows for spectral clustering
    row_norms = np.linalg.norm(X, axis=1, keepdims=True)
    row_norms = np.clip(row_norms, 1e-8, None)
    X_normed = X / row_norms

    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    return kmeans.fit_predict(X_normed)


def compute_module_coherence(
    data: HeteroData,
    modules: np.ndarray,
) -> dict[int, float]:
    """Per-module semantic coherence (avg pairwise cosine similarity).

    Args:
        data: PyG HeteroData with x_semantic.
        modules: [n_nodes] module assignment array.

    Returns:
        Dict of module_id -> coherence score.
    """
    semantic = data["node"].x_semantic.numpy()
    norms = np.linalg.norm(semantic, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)
    normed = semantic / norms

    coherence = {}
    for mod_id in np.unique(modules):
        members = np.where(modules == mod_id)[0]
        if len(members) < 2:
            coherence[int(mod_id)] = 1.0
            continue
        # Pairwise cosine similarities
        mod_normed = normed[members]
        sim_matrix = mod_normed @ mod_normed.T
        # Upper triangle mean (exclude diagonal)
        n_m = len(members)
        mask = np.triu(np.ones((n_m, n_m), dtype=bool), k=1)
        coherence[int(mod_id)] = float(sim_matrix[mask].mean())

    return coherence


def compute_phase2_results(data: HeteroData) -> dict:
    """Compute all Phase 2 proxy metrics for a single graph.

    Returns:
        Dict with local_variation, modules, module_coherence, node_ids.
    """
    local_var = compute_local_variation(data)
    modules = compute_module_assignments(data)
    coherence = compute_module_coherence(data, modules)

    return {
        "local_variation": local_var,
        "modules": modules,
        "module_coherence": coherence,
        "node_ids": data.node_ids if hasattr(data, "node_ids") else list(range(data["node"].num_nodes)),
    }
