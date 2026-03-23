"""Tier 3: Synthetic perturbation test (manufactured ground truth).

The strongest self-supervised signal. We create artificial misplaced concerns
by reassigning nodes between modules, then test whether the model detects them.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from topo_eval.phase2_proxy import compute_module_assignments
from topo_eval.perturbation import perturb_graph, control_perturbation


TARGETS = {
    "perturbation_sensitivity_mean": 0.6,
    "perturbation_specificity_mean": 0.75,
    "perturbation_error_delta_mean": 0.05,
    "control_sensitivity_mean_max": 0.25,  # should be ~0.20±0.05
}


@torch.no_grad()
def _get_reconstruction_errors(model, data, device: torch.device) -> np.ndarray:
    """Compute per-node reconstruction error for a single graph."""
    loader = DataLoader([data], batch_size=1)
    batch = next(iter(loader)).to(device)

    mask = torch.zeros(batch["node"].num_nodes, dtype=torch.bool, device=device)
    z_str, *_ = model(batch, mask)
    pred = model.decode(z_str)
    target = batch["node"].x_semantic.float()

    errors = 1.0 - F.cosine_similarity(pred, target, dim=-1)
    return errors.cpu().numpy()


def tier3_perturbation_test(
    model,
    dataset: list,
    device: torch.device,
    n_trials: int = 5,
    swap_fraction: float = 0.08,
    seed: int = 42,
) -> dict:
    """Run synthetic perturbation test on a dataset.

    For each graph and trial:
    1. Assign nodes to modules (spectral clustering).
    2. Swap ~8% of nodes between modules, rewiring edges.
    3. Compute reconstruction errors before and after.
    4. Measure whether swapped nodes are detected (higher error).

    Also runs control perturbation (no-op) to verify the model isn't
    responding to noise.

    Args:
        model: Trained R-GIN in eval mode.
        dataset: List of PyG HeteroData graphs.
        device: Torch device.
        n_trials: Number of perturbation trials per graph.
        swap_fraction: Fraction of nodes to swap.
        seed: Base random seed.

    Returns:
        Dict with sensitivity, specificity, error delta, control stats.
    """
    model.eval()

    sensitivities = []
    specificities = []
    error_deltas = []
    precisions = []
    control_sensitivities = []

    per_repo = {}

    for data_idx, data in enumerate(dataset):
        repo_key = getattr(data, "repo", f"repo_{data_idx}")
        n = data["node"].num_nodes

        if n < 10:
            continue  # Too small for meaningful perturbation

        modules = compute_module_assignments(data)
        n_modules = len(np.unique(modules))
        if n_modules < 2:
            continue  # Need at least 2 modules to swap between

        repo_sens = []
        repo_spec = []
        repo_delta = []
        repo_prec = []

        # Original errors (baseline)
        errors_orig = _get_reconstruction_errors(model, data, device)

        for trial in range(n_trials):
            rng = np.random.default_rng(seed + data_idx * 1000 + trial)

            # Perturbed graph
            data_pert, swap_set, new_modules = perturb_graph(
                data, modules, swap_fraction=swap_fraction, rng=rng,
            )

            if len(swap_set) == 0:
                continue

            errors_pert = _get_reconstruction_errors(model, data_pert, device)

            all_nodes = set(range(n))
            nonswap_set = all_nodes - swap_set
            swap_list = sorted(swap_set)
            nonswap_list = sorted(nonswap_set)

            # Error increase for swapped vs non-swapped
            swap_delta = errors_pert[swap_list].mean() - errors_orig[swap_list].mean()
            nonswap_delta = errors_pert[nonswap_list].mean() - errors_orig[nonswap_list].mean()
            error_deltas.append(float(swap_delta - nonswap_delta))
            repo_delta.append(float(swap_delta - nonswap_delta))

            # Sensitivity: fraction of swapped nodes in top-20% error after perturbation
            # Use rank-based cutoff to handle ties correctly
            k_top = max(1, int(np.ceil(n * 0.2)))
            top_indices = set(np.argsort(errors_pert)[-k_top:])
            high_error = np.array([i in top_indices for i in range(n)])

            sensitivity = high_error[swap_list].mean()
            sensitivities.append(float(sensitivity))
            repo_sens.append(float(sensitivity))

            # Specificity: fraction of non-swapped nodes NOT in top-20% error
            specificity = (~high_error[nonswap_list]).mean() if nonswap_list else 1.0
            specificities.append(float(specificity))
            repo_spec.append(float(specificity))

            # Precision: of all nodes flagged (top-20%), what fraction are actually swapped?
            flagged = set(np.where(high_error)[0])
            true_pos = flagged & swap_set
            precision = len(true_pos) / len(flagged) if flagged else 0.0
            precisions.append(float(precision))
            repo_prec.append(float(precision))

        # Control perturbation (no-op relabeling)
        for trial in range(n_trials):
            rng = np.random.default_rng(seed + 99999 + data_idx * 1000 + trial)
            data_ctrl, ctrl_swap = control_perturbation(data, modules, swap_fraction, rng)

            if not ctrl_swap:
                continue

            errors_ctrl = _get_reconstruction_errors(model, data_ctrl, device)
            k_top_ctrl = max(1, int(np.ceil(n * 0.2)))
            top_ctrl_indices = set(np.argsort(errors_ctrl)[-k_top_ctrl:])
            high_ctrl = np.array([i in top_ctrl_indices for i in range(n)])
            ctrl_swap_list = sorted(ctrl_swap)
            ctrl_sens = high_ctrl[ctrl_swap_list].mean()
            control_sensitivities.append(float(ctrl_sens))

        per_repo[repo_key] = {
            "n_nodes": n,
            "n_modules": n_modules,
            "sensitivity_mean": float(np.mean(repo_sens)) if repo_sens else 0.0,
            "specificity_mean": float(np.mean(repo_spec)) if repo_spec else 0.0,
            "precision_mean": float(np.mean(repo_prec)) if repo_prec else 0.0,
            "error_delta_mean": float(np.mean(repo_delta)) if repo_delta else 0.0,
        }

    results = {
        "perturbation_sensitivity_mean": float(np.mean(sensitivities)) if sensitivities else 0.0,
        "perturbation_sensitivity_std": float(np.std(sensitivities)) if sensitivities else 0.0,
        "perturbation_specificity_mean": float(np.mean(specificities)) if specificities else 0.0,
        "perturbation_specificity_std": float(np.std(specificities)) if specificities else 0.0,
        "perturbation_precision_mean": float(np.mean(precisions)) if precisions else 0.0,
        "perturbation_precision_std": float(np.std(precisions)) if precisions else 0.0,
        "perturbation_error_delta_mean": float(np.mean(error_deltas)) if error_deltas else 0.0,
        "perturbation_error_delta_std": float(np.std(error_deltas)) if error_deltas else 0.0,
        "control_sensitivity_mean": float(np.mean(control_sensitivities)) if control_sensitivities else 0.0,
        "control_sensitivity_std": float(np.std(control_sensitivities)) if control_sensitivities else 0.0,
        "n_trials_total": len(sensitivities),
        "n_repos_evaluated": len(per_repo),
        "per_repo": per_repo,
    }

    # Pass/fail
    results["passes"] = {
        "sensitivity": results["perturbation_sensitivity_mean"] > TARGETS["perturbation_sensitivity_mean"],
        "specificity": results["perturbation_specificity_mean"] > TARGETS["perturbation_specificity_mean"],
        "error_delta": results["perturbation_error_delta_mean"] > TARGETS["perturbation_error_delta_mean"],
        "control": 0.15 <= results["control_sensitivity_mean"] <= TARGETS["control_sensitivity_mean_max"],
    }
    results["tier3_pass"] = all(results["passes"].values())

    return results
