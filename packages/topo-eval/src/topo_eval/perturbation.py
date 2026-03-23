"""Graph perturbation utilities for synthetic misplacement testing.

Creates artificial misplaced concerns by reassigning nodes between modules
and rewiring a subset of edges to reflect the new module assignment.
"""

import numpy as np
import torch
from torch_geometric.data import HeteroData


def perturb_graph(
    data: HeteroData,
    modules: np.ndarray,
    swap_fraction: float = 0.08,
    edge_rewire_fraction: float = 0.5,
    rng: np.random.Generator | None = None,
) -> tuple[HeteroData, set[int], np.ndarray]:
    """Create a perturbed graph by swapping nodes between modules and rewiring edges.

    Args:
        data: Original PyG HeteroData graph.
        modules: [n_nodes] module assignment array.
        swap_fraction: Fraction of nodes to swap (default 8%).
        edge_rewire_fraction: Fraction of old-module edges to remove (default 50%).
        rng: Numpy random generator for reproducibility.

    Returns:
        Tuple of:
            - data_perturbed: Modified HeteroData with rewired edges.
            - swap_set: Set of swapped node indices.
            - new_modules: Updated module assignments after swaps.
    """
    if rng is None:
        rng = np.random.default_rng()

    n = data["node"].num_nodes
    module_ids = list(np.unique(modules))

    if len(module_ids) < 2:
        # Can't swap if only one module
        return data.clone(), set(), modules.copy()

    # Select nodes to swap
    n_swap = max(2, int(n * swap_fraction))
    all_indices = np.arange(n)
    swap_indices = rng.choice(all_indices, size=min(n_swap, n), replace=False)
    swap_set = set(int(i) for i in swap_indices)

    # Assign each swapped node to a different module
    new_modules = modules.copy()
    for idx in swap_indices:
        old_mod = modules[idx]
        other_mods = [m for m in module_ids if m != old_mod]
        if other_mods:
            new_modules[idx] = rng.choice(other_mods)

    # Build module membership lookups
    old_members: dict[int, set[int]] = {}
    new_members: dict[int, set[int]] = {}
    for mod_id in module_ids:
        old_members[mod_id] = set(np.where(modules == mod_id)[0])
        new_members[mod_id] = set(np.where(new_modules == mod_id)[0])

    # Rewire edges for swapped nodes
    data_new = data.clone()
    for edge_type in ["calls", "imports", "inherits"]:
        key = ("node", edge_type, "node")
        if key not in data.edge_types:
            continue

        ei = data[key].edge_index.clone()
        if ei.shape[1] == 0:
            continue

        src_np = ei[0].numpy()
        tgt_np = ei[1].numpy()

        edges_to_keep = []
        edges_to_add_src = []
        edges_to_add_tgt = []

        for e_idx in range(ei.shape[1]):
            s, t = int(src_np[e_idx]), int(tgt_np[e_idx])

            # Check if either endpoint is a swapped node
            s_swapped = s in swap_set
            t_swapped = t in swap_set

            if not s_swapped and not t_swapped:
                # Neither endpoint swapped — keep edge unchanged
                edges_to_keep.append(e_idx)
                continue

            # For swapped nodes: remove edge to old module with probability edge_rewire_fraction
            # Check both directions — if both endpoints are swapped, both checks apply
            should_remove = False
            if s_swapped and t in old_members.get(modules[s], set()):
                should_remove = rng.random() < edge_rewire_fraction
            if not should_remove and t_swapped and s in old_members.get(modules[t], set()):
                should_remove = rng.random() < edge_rewire_fraction

            if not should_remove:
                edges_to_keep.append(e_idx)

        # Add new edges from swapped nodes to their new module members
        for idx in swap_indices:
            idx = int(idx)
            new_mod = new_modules[idx]
            new_mod_members = list(new_members[new_mod] - {idx})
            if not new_mod_members:
                continue

            # Add edges to ~30% of new module members (reasonable connectivity)
            n_new_edges = max(1, int(len(new_mod_members) * 0.3))
            targets = rng.choice(new_mod_members, size=min(n_new_edges, len(new_mod_members)), replace=False)
            for t in targets:
                edges_to_add_src.append(idx)
                edges_to_add_tgt.append(int(t))

        # Reconstruct edge index
        kept = ei[:, edges_to_keep] if edges_to_keep else torch.zeros(2, 0, dtype=torch.long)
        if edges_to_add_src:
            added = torch.tensor(
                [edges_to_add_src, edges_to_add_tgt], dtype=torch.long
            )
            new_ei = torch.cat([kept, added], dim=1)
        else:
            new_ei = kept

        data_new[key].edge_index = new_ei

    return data_new, swap_set, new_modules


def control_perturbation(
    data: HeteroData,
    modules: np.ndarray,
    swap_fraction: float = 0.08,
    rng: np.random.Generator | None = None,
) -> tuple[HeteroData, set[int]]:
    """Control perturbation: 'swap' nodes to the SAME module (no-op relabeling).

    The graph structure is unchanged. If the model flags these nodes,
    it's responding to noise, not structural change.

    Args:
        data: Original graph.
        modules: Module assignments.
        swap_fraction: Fraction of nodes selected (but not actually moved).
        rng: Random generator.

    Returns:
        Tuple of (unchanged data clone, set of 'swapped' node indices).
    """
    if rng is None:
        rng = np.random.default_rng()

    n = data["node"].num_nodes
    n_swap = max(2, int(n * swap_fraction))
    swap_indices = rng.choice(n, size=min(n_swap, n), replace=False)
    swap_set = set(int(i) for i in swap_indices)

    return data.clone(), swap_set
