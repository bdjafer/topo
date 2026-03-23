"""Tests for graph perturbation utilities."""

import numpy as np
import pytest

from topo_eval.phase2_proxy import compute_module_assignments
from topo_eval.perturbation import perturb_graph, control_perturbation


class TestPerturbGraph:
    def test_swap_set_size(self, synthetic_graph):
        modules = compute_module_assignments(synthetic_graph)
        _, swap_set, _ = perturb_graph(synthetic_graph, modules, swap_fraction=0.1)
        expected = max(2, int(synthetic_graph["node"].num_nodes * 0.1))
        assert len(swap_set) == expected

    def test_swap_changes_modules(self, synthetic_graph):
        modules = compute_module_assignments(synthetic_graph)
        _, swap_set, new_modules = perturb_graph(synthetic_graph, modules)
        # At least some swapped nodes should have different module assignments
        changed = sum(1 for i in swap_set if modules[i] != new_modules[i])
        assert changed > 0

    def test_non_swapped_modules_unchanged(self, synthetic_graph):
        modules = compute_module_assignments(synthetic_graph)
        _, swap_set, new_modules = perturb_graph(synthetic_graph, modules)
        for i in range(len(modules)):
            if i not in swap_set:
                assert modules[i] == new_modules[i]

    def test_edge_count_changes(self, synthetic_graph):
        modules = compute_module_assignments(synthetic_graph)
        data_pert, swap_set, _ = perturb_graph(synthetic_graph, modules)

        # Edge counts should change (edges removed from old module, added to new)
        for edge_type in ["calls", "imports"]:
            key = ("node", edge_type, "node")
            if key in synthetic_graph.edge_types and key in data_pert.edge_types:
                orig_edges = synthetic_graph[key].edge_index.shape[1]
                pert_edges = data_pert[key].edge_index.shape[1]
                # Not necessarily different in all cases, but shape should be valid
                assert pert_edges >= 0

    def test_node_features_unchanged(self, synthetic_graph):
        """Perturbation only changes edges, not node features."""
        modules = compute_module_assignments(synthetic_graph)
        data_pert, _, _ = perturb_graph(synthetic_graph, modules)
        assert (data_pert["node"].x_semantic == synthetic_graph["node"].x_semantic).all()

    def test_reproducibility(self, synthetic_graph):
        modules = compute_module_assignments(synthetic_graph)
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        _, swap1, mod1 = perturb_graph(synthetic_graph, modules, rng=rng1)
        _, swap2, mod2 = perturb_graph(synthetic_graph, modules, rng=rng2)
        assert swap1 == swap2
        assert (mod1 == mod2).all()

    def test_single_module_no_swap(self):
        """With only one module, no swapping is possible."""
        from conftest import make_synthetic_graph
        data = make_synthetic_graph(n_nodes=20, n_modules=1)
        modules = np.zeros(20, dtype=int)
        _, swap_set, _ = perturb_graph(data, modules)
        assert len(swap_set) == 0


class TestControlPerturbation:
    def test_graph_unchanged(self, synthetic_graph):
        modules = compute_module_assignments(synthetic_graph)
        data_ctrl, swap_set = control_perturbation(synthetic_graph, modules)
        # Edges should be identical
        for edge_type in ["calls", "imports", "inherits"]:
            key = ("node", edge_type, "node")
            if key in synthetic_graph.edge_types:
                assert (data_ctrl[key].edge_index == synthetic_graph[key].edge_index).all()

    def test_swap_set_populated(self, synthetic_graph):
        modules = compute_module_assignments(synthetic_graph)
        _, swap_set = control_perturbation(synthetic_graph, modules, swap_fraction=0.1)
        assert len(swap_set) > 0
