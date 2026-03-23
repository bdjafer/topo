"""Test fixtures for topo-model."""

import torch
import pytest
from torch_geometric.data import HeteroData, Batch

from topo_model.config import RGINConfig


@pytest.fixture
def config():
    """Small config for fast testing."""
    return RGINConfig(
        hidden_dim=64,
        n_layers=2,
        n_node_types=5,
        dropout=0.0,  # deterministic for tests
        spectral_k=16,
        signnet_hidden=32,
        signnet_out_per_eig=2,
        invariant_dim=32,
        per_relation_dim=16,
    )


def _make_graph(n: int = 20, seed: int = 0) -> HeteroData:
    """Create a synthetic HeteroData graph for testing."""
    torch.manual_seed(seed)
    data = HeteroData()

    data["node"].x_semantic = torch.randn(n, 768)
    data["node"].x_spectral_vecs = torch.randn(n, 16)
    data["node"].x_spectral_vals = torch.rand(n, 16)
    data["node"].x_rwpe = torch.randn(n, 16)
    data["node"].x_tree = torch.rand(n, 4) * 10  # raw tree features
    data["node"].x_type = torch.randint(0, 5, (n,))
    data["node"].num_nodes = n

    # Add edges: calls (chain), imports (some), inherits (few)
    calls_src = list(range(n - 1))
    calls_tgt = list(range(1, n))
    data["node", "calls", "node"].edge_index = torch.tensor(
        [calls_src, calls_tgt], dtype=torch.long
    )

    imports_src = list(range(0, n - 2, 2))
    imports_tgt = list(range(1, n - 1, 2))
    data["node", "imports", "node"].edge_index = torch.tensor(
        [imports_src, imports_tgt], dtype=torch.long
    )

    inherits_src = [0, 2, 4][:min(3, n // 4)]
    inherits_tgt = [1, 3, 5][:min(3, n // 4)]
    if inherits_src:
        data["node", "inherits", "node"].edge_index = torch.tensor(
            [inherits_src, inherits_tgt], dtype=torch.long
        )
    else:
        data["node", "inherits", "node"].edge_index = torch.zeros(2, 0, dtype=torch.long)

    return data


@pytest.fixture
def single_graph():
    """Single synthetic graph with 20 nodes."""
    return _make_graph(n=20, seed=0)


@pytest.fixture
def batch_of_graphs():
    """Batch of 3 synthetic graphs."""
    graphs = [_make_graph(n=n, seed=i) for i, n in enumerate([20, 15, 25])]
    return Batch.from_data_list(graphs)
