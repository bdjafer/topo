"""Ablation baselines for comparison against R-GIN.

Phase 3 must outperform these naive baselines on the synthetic perturbation test.

Baselines:
1. Random: random anomaly scores (no graph structure)
2. Phase 2 local variation: semantic disagreement with neighbors
3. Centroid distance: cosine distance to module centroid
4. Degree-only: normalized degree (higher degree = higher score)
"""

import numpy as np

from topo_eval.phase2_proxy import compute_module_assignments, compute_local_variation
from topo_eval.perturbation import perturb_graph


def _degree_per_node_normalized(data) -> np.ndarray:
    """Compute normalized degree per node across all edge types.

    Returns values in [0, 1] by dividing by max degree (or 1 if all zero).
    """
    n = data["node"].num_nodes
    degree = np.zeros(n, dtype=np.float64)
    for edge_type in ["calls", "imports", "inherits"]:
        key = ("node", edge_type, "node")
        if key in data.edge_types:
            ei = data[key].edge_index.numpy()
            for j in range(ei.shape[1]):
                degree[ei[0, j]] += 1
                degree[ei[1, j]] += 1
    max_deg = degree.max()
    if max_deg > 0:
        degree = degree / max_deg
    return degree


def _centroid_distance(data, modules: np.ndarray) -> np.ndarray:
    """Per-node cosine distance to its module's semantic centroid."""
    semantic = data["node"].x_semantic.numpy()
    n = data["node"].num_nodes
    distances = np.zeros(n, dtype=np.float64)

    for mod_id in np.unique(modules):
        members = np.where(modules == mod_id)[0]
        if len(members) == 0:
            continue
        centroid = semantic[members].mean(axis=0)
        c_norm = np.linalg.norm(centroid)
        if c_norm < 1e-8:
            continue
        centroid_normed = centroid / c_norm
        for idx in members:
            v_norm = np.linalg.norm(semantic[idx])
            if v_norm < 1e-8:
                distances[idx] = 1.0
            else:
                distances[idx] = 1.0 - float(np.dot(semantic[idx] / v_norm, centroid_normed))

    return distances


def _evaluate_baseline_on_perturbation(
    score_fn,
    dataset: list,
    n_trials: int = 5,
    swap_fraction: float = 0.08,
    seed: int = 42,
) -> dict:
    """Run a baseline scorer through the same perturbation test protocol.

    Args:
        score_fn: Callable(data, modules) -> np.ndarray of per-node anomaly scores.
            The modules argument provides module assignments for contextual baselines.
        dataset: List of PyG HeteroData graphs.
        n_trials: Perturbation trials per graph.
        swap_fraction: Fraction of nodes to swap.
        seed: Random seed.

    Returns:
        Dict with sensitivity, specificity, precision, error_delta.
    """
    sensitivities = []
    specificities = []
    precisions = []
    error_deltas = []

    for data_idx, data in enumerate(dataset):
        n = data["node"].num_nodes
        if n < 10:
            continue

        modules = compute_module_assignments(data)
        if len(np.unique(modules)) < 2:
            continue

        scores_orig = score_fn(data, modules)

        for trial in range(n_trials):
            rng = np.random.default_rng(seed + data_idx * 1000 + trial)

            data_pert, swap_set, new_modules = perturb_graph(
                data, modules, swap_fraction=swap_fraction, rng=rng,
            )

            if not swap_set:
                continue

            # Score the perturbed graph with the NEW module assignments
            scores_pert = score_fn(data_pert, new_modules)

            all_nodes = set(range(n))
            nonswap_set = all_nodes - swap_set
            swap_list = sorted(swap_set)
            nonswap_list = sorted(nonswap_set)

            # Error delta
            swap_delta = scores_pert[swap_list].mean() - scores_orig[swap_list].mean()
            nonswap_delta = scores_pert[nonswap_list].mean() - scores_orig[nonswap_list].mean()
            error_deltas.append(float(swap_delta - nonswap_delta))

            # Sensitivity/specificity/precision (rank-based top-k for tie handling)
            k_top = max(1, int(np.ceil(n * 0.2)))
            top_indices = set(np.argsort(scores_pert)[-k_top:])
            high = np.array([i in top_indices for i in range(n)])

            sensitivity = high[swap_list].mean()
            sensitivities.append(float(sensitivity))

            specificity = (~high[nonswap_list]).mean() if nonswap_list else 1.0
            specificities.append(float(specificity))

            flagged = set(np.where(high)[0])
            true_pos = flagged & swap_set
            precision = len(true_pos) / len(flagged) if flagged else 0.0
            precisions.append(float(precision))

    return {
        "sensitivity_mean": float(np.mean(sensitivities)) if sensitivities else 0.0,
        "specificity_mean": float(np.mean(specificities)) if specificities else 0.0,
        "precision_mean": float(np.mean(precisions)) if precisions else 0.0,
        "error_delta_mean": float(np.mean(error_deltas)) if error_deltas else 0.0,
        "n_trials": len(sensitivities),
    }


def run_baselines(
    dataset: list,
    n_trials: int = 5,
    swap_fraction: float = 0.08,
    seed: int = 42,
) -> dict:
    """Run all ablation baselines through the perturbation test.

    Args:
        dataset: List of PyG HeteroData graphs.
        n_trials: Perturbation trials per graph.
        swap_fraction: Fraction of nodes to swap.
        seed: Random seed.

    Returns:
        Dict with results per baseline.
    """
    results = {}

    # Baseline 1: Random scores (independent RNG per call)
    _random_call_counter = [0]

    def random_scorer(data, modules):
        _random_call_counter[0] += 1
        rng = np.random.default_rng(seed + 777 + _random_call_counter[0])
        return rng.random(data["node"].num_nodes)

    results["random"] = _evaluate_baseline_on_perturbation(
        random_scorer, dataset, n_trials, swap_fraction, seed,
    )

    # Baseline 2: Phase 2 local variation (ignores modules argument)
    def local_var_scorer(data, modules):
        return compute_local_variation(data)

    results["phase2_local_variation"] = _evaluate_baseline_on_perturbation(
        local_var_scorer, dataset, n_trials, swap_fraction, seed,
    )

    # Baseline 3: Centroid distance (uses provided module assignments)
    # For original graph: scored with original modules
    # For perturbed graph: scored with NEW modules (post-swap)
    # This tests whether swapped nodes are far from their new module's centroid
    def centroid_scorer(data, modules):
        return _centroid_distance(data, modules)

    results["centroid_distance"] = _evaluate_baseline_on_perturbation(
        centroid_scorer, dataset, n_trials, swap_fraction, seed,
    )

    # Baseline 4: Degree-only (normalized to [0,1])
    def degree_scorer(data, modules):
        return _degree_per_node_normalized(data)

    results["degree_only"] = _evaluate_baseline_on_perturbation(
        degree_scorer, dataset, n_trials, swap_fraction, seed,
    )

    return results
