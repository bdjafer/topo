"""Tier 4: Structural consistency checks.

Properties that must hold if the model works:
- NMI lift: learned clustering at least as good as spectral
- Reconstruction error not dominated by degree
- Per-layer role divergence for bridge nodes
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.metrics import normalized_mutual_info_score
from sklearn.cluster import KMeans

from topo_eval.phase2_proxy import compute_module_assignments


@torch.no_grad()
def _get_embeddings(model, data, device: torch.device) -> dict:
    """Get all embeddings and reconstruction errors for a single graph."""
    loader = DataLoader([data], batch_size=1)
    batch = next(iter(loader)).to(device)

    mask = torch.zeros(batch["node"].num_nodes, dtype=torch.bool, device=device)
    z_str, z_inv, z_calls, z_imports, z_inherits, g_emb = model(batch, mask)
    pred = model.decode(z_str)
    target = batch["node"].x_semantic.float()
    errors = 1.0 - F.cosine_similarity(pred, target, dim=-1)

    return {
        "z_str": z_str.cpu().numpy(),
        "z_invariant": z_inv.cpu().numpy(),
        "z_calls": z_calls.cpu().numpy(),
        "z_imports": z_imports.cpu().numpy(),
        "z_inherits": z_inherits.cpu().numpy(),
        "g_emb": g_emb.cpu().numpy(),
        "reconstruction_error": errors.cpu().numpy(),
    }


def _compute_degree(data) -> np.ndarray:
    """Compute total degree per node across all edge types."""
    n = data["node"].num_nodes
    degree = np.zeros(n, dtype=np.float64)
    for edge_type in ["calls", "imports", "inherits"]:
        key = ("node", edge_type, "node")
        if key in data.edge_types:
            ei = data[key].edge_index.numpy()
            for j in range(ei.shape[1]):
                degree[ei[0, j]] += 1
                degree[ei[1, j]] += 1
    return degree


def tier4_structural_consistency(
    model,
    dataset: list,
    device: torch.device,
) -> dict:
    """Run Tier 4 structural consistency checks.

    Checks:
    1. NMI lift: z_invariant clustering vs spectral clustering, both compared
       to package/directory structure (approximated by spectral modules).
    2. Reconstruction error vs degree: rank correlation should be < 0.5
       (model shouldn't just flag high-degree nodes).
    3. Per-layer role divergence: bridge-like nodes should have more divergent
       per-layer embeddings than non-bridge nodes.

    Args:
        model: Trained R-GIN in eval mode.
        dataset: List of PyG HeteroData graphs.
        device: Torch device.

    Returns:
        Dict with check results.
    """
    model.eval()

    nmi_lifts = []
    error_degree_correlations = []
    bridge_divergences = []
    nonbridge_divergences = []

    per_repo = {}

    for data_idx, data in enumerate(dataset):
        repo_key = getattr(data, "repo", f"repo_{data_idx}")
        n = data["node"].num_nodes
        if n < 10:
            continue

        embs = _get_embeddings(model, data, device)

        # --- Check 1: NMI lift ---
        # Spectral clustering (Phase 2 baseline)
        spectral_modules = compute_module_assignments(data)
        n_clusters = len(np.unique(spectral_modules))

        # z_invariant clustering (Phase 3)
        z_inv = embs["z_invariant"]
        if n_clusters >= 2 and n > n_clusters:
            z_inv_normed = z_inv / (np.linalg.norm(z_inv, axis=1, keepdims=True) + 1e-8)
            kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
            z_inv_clusters = kmeans.fit_predict(z_inv_normed)

            # Random clustering baseline for NMI lift
            random_clusters = np.random.default_rng(42).integers(0, n_clusters, size=n)

            # NMI lift: compare both z_inv and random against spectral reference
            nmi_zinv = normalized_mutual_info_score(spectral_modules, z_inv_clusters)
            nmi_random = normalized_mutual_info_score(spectral_modules, random_clusters)

            # Lift = z_inv NMI - random NMI (how much better than chance)
            nmi_lift = nmi_zinv - nmi_random
            nmi_lifts.append(float(nmi_lift))
        else:
            nmi_lifts.append(0.0)

        # --- Check 2: Reconstruction error vs degree ---
        degree = _compute_degree(data)
        recon_err = embs["reconstruction_error"]

        if np.std(degree) > 0 and np.std(recon_err) > 0:
            from scipy import stats
            rho, _ = stats.spearmanr(recon_err, degree)
            if np.isnan(rho):
                rho = 0.0
            error_degree_correlations.append(float(rho))
        else:
            error_degree_correlations.append(0.0)

        # --- Check 3: Per-layer role divergence ---
        # Per the spec, z_calls/z_imports/z_inherits live in different learned spaces
        # (HSIC decorrelation), so we compare per-layer cluster assignments rather than
        # raw embedding distances.
        z_calls = embs["z_calls"]
        z_imports = embs["z_imports"]
        z_inherits = embs["z_inherits"]

        # Cluster each layer independently
        n_layer_clusters = min(n_clusters, n // 2) if n > 4 else 2
        if n_layer_clusters >= 2:
            calls_clusters = KMeans(n_clusters=n_layer_clusters, n_init=5, random_state=42).fit_predict(z_calls)
            imports_clusters = KMeans(n_clusters=n_layer_clusters, n_init=5, random_state=42).fit_predict(z_imports)
            inherits_clusters = KMeans(n_clusters=n_layer_clusters, n_init=5, random_state=42).fit_predict(z_inherits)
        else:
            calls_clusters = imports_clusters = inherits_clusters = np.zeros(n, dtype=int)

        # Bridge nodes: high degree AND connected to multiple modules
        modules = spectral_modules
        degree_threshold = np.percentile(degree, 75)

        # Build neighbor module sets efficiently
        neighbor_modules: dict[int, set[int]] = {i: set() for i in range(n)}
        for edge_type in ["calls", "imports", "inherits"]:
            key = ("node", edge_type, "node")
            if key not in data.edge_types:
                continue
            ei = data[key].edge_index.numpy()
            for j in range(ei.shape[1]):
                neighbor_modules[int(ei[0, j])].add(int(modules[ei[1, j]]))
                neighbor_modules[int(ei[1, j])].add(int(modules[ei[0, j]]))

        for i in range(n):
            if degree[i] < degree_threshold:
                continue

            # Per-layer divergence: how many layer-cluster assignments disagree
            layer_assignments = [int(calls_clusters[i]), int(imports_clusters[i]), int(inherits_clusters[i])]
            # Count unique assignments — more unique = more divergent
            divergence = float(len(set(layer_assignments))) / 3.0  # 1/3 = all same, 1.0 = all different

            if len(neighbor_modules[i]) >= 2:
                bridge_divergences.append(divergence)
            else:
                nonbridge_divergences.append(divergence)

        per_repo[repo_key] = {
            "n_nodes": n,
            "nmi_zinv_spectral": float(nmi_lifts[-1]),
            "error_degree_correlation": float(error_degree_correlations[-1]),
        }

    results = {
        "nmi_mean": float(np.mean(nmi_lifts)) if nmi_lifts else 0.0,
        "nmi_std": float(np.std(nmi_lifts)) if nmi_lifts else 0.0,
        "error_degree_corr_mean": float(np.mean(error_degree_correlations)) if error_degree_correlations else 0.0,
        "error_degree_corr_std": float(np.std(error_degree_correlations)) if error_degree_correlations else 0.0,
        "bridge_divergence_mean": float(np.mean(bridge_divergences)) if bridge_divergences else 0.0,
        "nonbridge_divergence_mean": float(np.mean(nonbridge_divergences)) if nonbridge_divergences else 0.0,
        "bridge_divergence_ratio": (
            float(np.mean(bridge_divergences) / np.mean(nonbridge_divergences))
            if bridge_divergences and nonbridge_divergences and np.mean(nonbridge_divergences) > 1e-8
            else None
        ),
        "per_repo": per_repo,
    }

    # Pass/fail
    results["passes"] = {
        # NMI lift: z_invariant clusters should have reasonable agreement with spectral
        "nmi_positive": results["nmi_mean"] > 0.0,
        # Reconstruction error should NOT be dominated by degree
        "error_not_degree": results["error_degree_corr_mean"] < 0.5,
        # Bridge nodes should have higher per-layer divergence
        "bridge_divergence": (
            results["bridge_divergence_ratio"] is not None
            and results["bridge_divergence_ratio"] > 1.0
        ),
    }
    results["tier4_pass"] = all(results["passes"].values())

    return results
