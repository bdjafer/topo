"""Tests for data augmentation transforms."""

import torch
import pytest

from topo_dataset.loader import load_graph
from topo_dataset.transforms import sample_subgraph, _induce_subgraph


class TestSampleSubgraph:
    def test_ratio_respected(self, synthetic_repo):
        data = load_graph(synthetic_repo)
        sub = sample_subgraph(data, ratio=0.5, seed=42)
        n_sub = sub["node"].x_type.size(0)
        # Should be approximately 50% of 10 = 5 nodes
        assert 3 <= n_sub <= 7

    def test_deterministic_with_seed(self, synthetic_repo):
        data = load_graph(synthetic_repo)
        sub1 = sample_subgraph(data, ratio=0.7, seed=123)
        sub2 = sample_subgraph(data, ratio=0.7, seed=123)
        assert torch.equal(sub1["node"].x_type, sub2["node"].x_type)

    def test_multiple_seeds_all_valid(self, synthetic_repo):
        """Multiple seeds all produce valid, non-empty subgraphs."""
        data = load_graph(synthetic_repo)
        for seed in [0, 1, 42, 999]:
            sub = sample_subgraph(data, ratio=0.5, seed=seed)
            n_sub = sub["node"].x_type.size(0)
            assert n_sub >= 1, f"seed={seed} produced empty subgraph"
            assert sub["node"].x_semantic.shape == (n_sub, 768)

    def test_full_ratio_reaches_component(self, synthetic_repo):
        """ratio=1.0 reaches all nodes in the start node's connected component.

        The synthetic graph has orphan nodes (7,8,9) unreachable by BFS,
        so we can't guarantee all 10 nodes — only the connected component.
        """
        data = load_graph(synthetic_repo)
        sub = sample_subgraph(data, ratio=1.0, seed=42)
        n_sub = sub["node"].x_type.size(0)
        # Should reach at least the main component (nodes 0-6 = 7 nodes)
        assert n_sub >= 7

    def test_node_features_preserved(self, synthetic_repo):
        data = load_graph(synthetic_repo)
        sub = sample_subgraph(data, ratio=0.7, seed=42)

        # All feature tensors should have consistent first dimension
        n_sub = sub["node"].x_type.size(0)
        assert sub["node"].x_semantic.shape[0] == n_sub
        assert sub["node"].x_spectral_vecs.shape[0] == n_sub
        assert sub["node"].x_rwpe.shape[0] == n_sub
        assert sub["node"].x_tree.shape[0] == n_sub

    def test_edge_indices_valid(self, synthetic_repo):
        data = load_graph(synthetic_repo)
        sub = sample_subgraph(data, ratio=0.7, seed=42)
        n_sub = sub["node"].x_type.size(0)

        for edge_type in ["calls", "imports", "inherits"]:
            key = ("node", edge_type, "node")
            if key in sub.edge_types:
                edges = sub[key].edge_index
                assert edges.min() >= 0
                assert edges.max() < n_sub

    def test_no_inherits_repo(self, synthetic_repo_no_inherits):
        data = load_graph(synthetic_repo_no_inherits)
        sub = sample_subgraph(data, ratio=0.7, seed=42)
        assert sub["node"].x_type.size(0) >= 1
        # inherits should not be present in subgraph either
        assert ("node", "inherits", "node") not in sub.edge_types


class TestInduceSubgraph:
    def test_identity(self, synthetic_repo):
        """Inducing with all nodes = identity."""
        data = load_graph(synthetic_repo)
        subset = torch.arange(10)
        sub = _induce_subgraph(data, subset)
        assert sub["node"].x_type.size(0) == 10
        assert torch.equal(sub["node"].x_type, data["node"].x_type)

    def test_single_node(self, synthetic_repo):
        """Inducing with one node should drop all edges."""
        data = load_graph(synthetic_repo)
        subset = torch.tensor([5])
        sub = _induce_subgraph(data, subset)
        assert sub["node"].x_type.size(0) == 1
        # Node 5 is an inherits source — but since it's the only node,
        # its edges to node 2 are dropped
        for edge_type in ["calls", "imports", "inherits"]:
            key = ("node", edge_type, "node")
            if key in sub.edge_types:
                assert sub[key].edge_index.shape[1] == 0

    def test_edge_reindexing(self, synthetic_repo):
        """Edge indices should be re-indexed to [0, n_sub)."""
        data = load_graph(synthetic_repo)
        # Nodes 0,1,2,3 have calls edges: 0->1, 1->2, 2->3
        subset = torch.tensor([0, 1, 2, 3])
        sub = _induce_subgraph(data, subset)

        calls = sub["node", "calls", "node"].edge_index
        # Original: [[0,1,2,3],[1,2,3,4]] — node 4 is excluded
        # After subset [0,1,2,3]: edges 0->1, 1->2, 2->3 survive
        assert calls.shape[1] == 3  # 3 edges survive (4th goes to node 4)
        assert calls.max() < 4
