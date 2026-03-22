"""Data augmentation transforms for R-GIN training.

Subgraph sampling for graph contrastive loss (Loss 3 in PHASE_3.md).
"""

from typing import Optional

import torch
from torch_geometric.data import HeteroData


def sample_subgraph(
    data: HeteroData,
    ratio: float = 0.7,
    seed: Optional[int] = None,
) -> HeteroData:
    """Sample a connected subgraph via BFS from a random start node.

    Args:
        data: Full graph as HeteroData.
        ratio: Fraction of nodes to include (0.0, 1.0].
        seed: Optional random seed for reproducibility.

    Returns:
        Induced subgraph as a new HeteroData (with re-indexed edges).
    """
    n = data["node"].num_nodes
    target_size = max(1, int(n * ratio))

    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(seed)
    start = torch.randint(0, n, (1,), generator=gen).item()

    # Build undirected adjacency for BFS using tensor ops
    all_src = []
    all_dst = []
    for edge_type in ["calls", "imports", "inherits"]:
        edge_key = ("node", edge_type, "node")
        if edge_key not in data.edge_types:
            continue
        edges = data[edge_key].edge_index
        all_src.append(edges[0])
        all_dst.append(edges[1])

    # Build neighbor sets from concatenated edges
    neighbors: dict[int, list[int]] = {}
    if all_src:
        src_cat = torch.cat(all_src)
        dst_cat = torch.cat(all_dst)
        # Symmetrize: add both directions
        full_src = torch.cat([src_cat, dst_cat]).tolist()
        full_dst = torch.cat([dst_cat, src_cat]).tolist()
        for s, d in zip(full_src, full_dst):
            if s not in neighbors:
                neighbors[s] = []
            neighbors[s].append(d)

    # BFS expansion
    visited: set[int] = {start}
    frontier = [start]
    while len(visited) < target_size and frontier:
        next_frontier = []
        for node in frontier:
            for nb in neighbors.get(node, []):
                if nb not in visited and len(visited) < target_size:
                    visited.add(nb)
                    next_frontier.append(nb)
        frontier = next_frontier

    # Induce subgraph
    subset = torch.tensor(sorted(visited), dtype=torch.long)
    return _induce_subgraph(data, subset)


def _induce_subgraph(data: HeteroData, subset: torch.Tensor) -> HeteroData:
    """Induce a subgraph on the given node subset.

    Re-indexes edges so that node indices are contiguous [0, len(subset)).
    """
    sub = HeteroData()
    n_sub = subset.size(0)
    n_orig = data["node"].num_nodes

    # Node index mapping: old -> new
    mapping = torch.full((n_orig,), -1, dtype=torch.long)
    mapping[subset] = torch.arange(n_sub, dtype=torch.long)

    # Copy node features
    for key in data["node"].keys():
        if key == "num_nodes":
            continue
        attr = data["node"][key]
        if isinstance(attr, torch.Tensor) and attr.size(0) == n_orig:
            sub["node"][key] = attr[subset]
    sub["node"].num_nodes = n_sub

    # Copy and re-index edges
    for edge_type in ["calls", "imports", "inherits"]:
        edge_key = ("node", edge_type, "node")
        if edge_key not in data.edge_types:
            continue
        edges = data[edge_key].edge_index  # [2, m]
        src, dst = edges[0], edges[1]

        # Keep only edges where both endpoints are in subset
        src_mapped = mapping[src]
        dst_mapped = mapping[dst]
        mask = (src_mapped >= 0) & (dst_mapped >= 0)

        if mask.any():
            sub["node", edge_type, "node"].edge_index = torch.stack(
                [src_mapped[mask], dst_mapped[mask]]
            )

    return sub
