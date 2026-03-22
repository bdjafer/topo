"""Tests for NPZ -> PyG HeteroData loader."""

import numpy as np
import torch
import pytest

from topo_dataset.loader import load_graph


class TestLoadGraph:
    def test_loads_all_node_features(self, synthetic_repo):
        data = load_graph(synthetic_repo)

        assert data["node"].x_semantic.shape == (10, 768)
        assert data["node"].x_spectral_vecs.shape == (10, 16)
        assert data["node"].x_spectral_vals.shape == (10, 16)
        assert data["node"].x_rwpe.shape == (10, 16)
        assert data["node"].x_tree.shape == (10, 4)
        assert data["node"].x_type.shape == (10,)

    def test_dtypes(self, synthetic_repo):
        data = load_graph(synthetic_repo)

        assert data["node"].x_semantic.dtype == torch.float32
        assert data["node"].x_spectral_vecs.dtype == torch.float32
        assert data["node"].x_spectral_vals.dtype == torch.float32
        assert data["node"].x_rwpe.dtype == torch.float32
        assert data["node"].x_tree.dtype == torch.float32
        assert data["node"].x_type.dtype == torch.int64

    def test_tree_features_log_compressed(self, synthetic_repo):
        """Tree features should be log1p-compressed."""
        data = load_graph(synthetic_repo)

        # Root node has [0, 0, 10, 0] raw -> log1p -> [0, 0, log(11), 0]
        expected_subtree = torch.log1p(torch.tensor(10.0))
        assert torch.isclose(data["node"].x_tree[0, 2], expected_subtree)

        # Zero stays zero after log1p
        assert data["node"].x_tree[0, 0] == 0.0

    def test_edge_indices(self, synthetic_repo):
        data = load_graph(synthetic_repo)

        assert ("node", "calls", "node") in data.edge_types
        assert ("node", "imports", "node") in data.edge_types
        assert ("node", "inherits", "node") in data.edge_types

        calls = data["node", "calls", "node"].edge_index
        assert calls.shape == (2, 4)
        assert calls.dtype == torch.int64

    def test_empty_edge_type_excluded(self, synthetic_repo_no_inherits):
        """Repos with no inherits edges should not have that edge type."""
        data = load_graph(synthetic_repo_no_inherits)

        assert ("node", "calls", "node") in data.edge_types
        assert ("node", "imports", "node") in data.edge_types
        # Empty inherits (shape [2,0]) should NOT create an edge type
        assert ("node", "inherits", "node") not in data.edge_types

    def test_metadata(self, synthetic_repo):
        data = load_graph(synthetic_repo)

        assert data.n_nodes == 10
        assert len(data.node_ids) == 10
        assert data.node_ids[0] == "node_0"
        assert data.graph_meta["fiedler_value"] == 0.42

    def test_no_nan_inf(self, synthetic_repo):
        """Loaded tensors should contain no NaN or Inf."""
        data = load_graph(synthetic_repo)

        for attr_name in ["x_semantic", "x_spectral_vecs", "x_spectral_vals", "x_rwpe", "x_tree"]:
            t = data["node"][attr_name]
            assert not torch.isnan(t).any(), f"NaN in {attr_name}"
            assert not torch.isinf(t).any(), f"Inf in {attr_name}"
