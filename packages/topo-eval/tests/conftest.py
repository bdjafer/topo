"""Shared fixtures for topo-eval tests."""

import numpy as np
import pytest
import torch
from torch_geometric.data import HeteroData

from topo_model.config import RGINConfig
from topo_model.rgin import RGIN


def make_synthetic_graph(
    n_nodes: int = 50,
    n_modules: int = 3,
    edge_density: float = 0.15,
    seed: int = 42,
) -> HeteroData:
    """Create a synthetic graph with known module structure.

    Nodes within the same module are densely connected;
    nodes across modules have sparse connections.
    """
    rng = np.random.default_rng(seed)

    data = HeteroData()

    # Assign nodes to modules
    module_sizes = [n_nodes // n_modules] * n_modules
    module_sizes[-1] += n_nodes - sum(module_sizes)
    modules = []
    for i, sz in enumerate(module_sizes):
        modules.extend([i] * sz)
    modules = np.array(modules)

    # Generate semantic embeddings with module-correlated structure
    centroids = rng.standard_normal((n_modules, 768)).astype(np.float32)
    centroids = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
    semantic = np.zeros((n_nodes, 768), dtype=np.float32)
    for i in range(n_nodes):
        noise = rng.standard_normal(768).astype(np.float32) * 0.3
        semantic[i] = centroids[modules[i]] + noise
        semantic[i] /= np.linalg.norm(semantic[i]) + 1e-8

    data["node"].x_semantic = torch.from_numpy(semantic)

    # Spectral PEs (mock)
    k = 16
    data["node"].x_spectral_vecs = torch.randn(n_nodes, k)
    data["node"].x_spectral_vals = torch.rand(n_nodes, k)

    # RWPE
    data["node"].x_rwpe = torch.randn(n_nodes, 16)

    # Tree features
    data["node"].x_tree = torch.rand(n_nodes, 4)

    # Node types
    data["node"].x_type = torch.randint(0, 5, (n_nodes,))

    # Edges: dense within modules, sparse across
    for edge_type in ["calls", "imports", "inherits"]:
        src_list, tgt_list = [], []
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                same_module = modules[i] == modules[j]
                prob = edge_density if same_module else edge_density * 0.1
                if rng.random() < prob:
                    src_list.append(i)
                    tgt_list.append(j)
        if src_list:
            ei = torch.tensor([src_list, tgt_list], dtype=torch.long)
            data["node", edge_type, "node"].edge_index = ei

    data["node"].num_nodes = n_nodes
    data.repo = f"synthetic_{seed}"
    data.node_ids = [f"node_{i}" for i in range(n_nodes)]
    data.n_nodes = n_nodes
    data.graph_meta = {"n_nodes": n_nodes, "modules": modules.tolist()}

    return data


@pytest.fixture
def synthetic_graph():
    """Single synthetic graph with 50 nodes, 3 modules."""
    return make_synthetic_graph(n_nodes=50, n_modules=3, seed=42)


@pytest.fixture
def synthetic_dataset():
    """Dataset of 4 synthetic graphs."""
    return [
        make_synthetic_graph(n_nodes=50, n_modules=3, seed=42),
        make_synthetic_graph(n_nodes=80, n_modules=4, seed=43),
        make_synthetic_graph(n_nodes=30, n_modules=2, seed=44),
        make_synthetic_graph(n_nodes=60, n_modules=3, seed=45),
    ]


@pytest.fixture
def model():
    """Small R-GIN model for testing."""
    config = RGINConfig(hidden_dim=64, n_layers=2)
    m = RGIN(config)
    m.eval()
    return m


@pytest.fixture
def device():
    return torch.device("cpu")
