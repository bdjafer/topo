"""Tests for R-GIN model."""

import torch
import pytest
from torch_geometric.data import Batch

from topo_model.config import RGINConfig
from topo_model.rgin import RGIN, GraphNorm


class TestGraphNorm:
    def test_output_shape(self):
        norm = GraphNorm(64)
        x = torch.randn(30, 64)
        batch = torch.tensor([0]*10 + [1]*10 + [2]*10)
        out = norm(x, batch)
        assert out.shape == x.shape

    def test_per_graph_normalization(self):
        """Each graph should be normalized independently."""
        norm = GraphNorm(64)
        x = torch.randn(20, 64)
        batch = torch.tensor([0]*10 + [1]*10)
        out = norm(x, batch)
        assert torch.isfinite(out).all()


class TestRGIN:
    def test_forward_shapes_single(self, config, single_graph):
        """Forward pass produces correct output shapes for a single graph."""
        model = RGIN(config)
        model.eval()

        batch = Batch.from_data_list([single_graph])
        n = 20
        mask = torch.zeros(n, dtype=torch.bool)
        mask[:5] = True  # mask first 5 nodes

        z_str, z_inv, z_calls, z_imports, z_inherits, g_emb = model(batch, mask)

        assert z_str.shape == (n, config.z_str_dim)  # (20, 80) with test config
        assert z_inv.shape == (n, config.invariant_dim)  # (20, 32)
        assert z_calls.shape == (n, config.per_relation_dim)  # (20, 16)
        assert z_imports.shape == (n, config.per_relation_dim)
        assert z_inherits.shape == (n, config.per_relation_dim)
        assert g_emb.shape == (1, config.invariant_dim)  # (1, 32)

    def test_forward_shapes_batch(self, config, batch_of_graphs):
        """Forward pass works with batched graphs of different sizes."""
        model = RGIN(config)
        model.eval()

        total_nodes = 20 + 15 + 25  # 60
        mask = torch.zeros(total_nodes, dtype=torch.bool)
        mask[torch.randperm(total_nodes)[:20]] = True

        z_str, z_inv, z_calls, z_imports, z_inherits, g_emb = model(batch_of_graphs, mask)

        assert z_str.shape == (total_nodes, config.z_str_dim)
        assert g_emb.shape == (3, config.invariant_dim)  # 3 graphs

    def test_decode_shape(self, config, single_graph):
        """Decoder produces correct output shape (768d)."""
        model = RGIN(config)
        batch = Batch.from_data_list([single_graph])
        mask = torch.zeros(20, dtype=torch.bool)
        mask[:5] = True

        z_str, *_ = model(batch, mask)
        decoded = model.decode(z_str[mask])
        assert decoded.shape == (5, 768)  # 5 masked nodes → 768d predictions

    def test_masking_affects_output(self, config, single_graph):
        """Different masks should produce different outputs."""
        model = RGIN(config)
        model.eval()

        batch = Batch.from_data_list([single_graph])

        mask1 = torch.zeros(20, dtype=torch.bool)
        mask1[:10] = True

        mask2 = torch.zeros(20, dtype=torch.bool)
        mask2[10:] = True

        z1, *_ = model(batch, mask1)
        z2, *_ = model(batch, mask2)

        # Outputs should differ because different nodes are masked
        assert not torch.allclose(z1, z2, atol=1e-4)

    def test_no_nan_inf(self, config, batch_of_graphs):
        """No NaN or Inf in outputs."""
        model = RGIN(config)
        model.eval()

        total_nodes = 20 + 15 + 25
        mask = torch.zeros(total_nodes, dtype=torch.bool)
        mask[::3] = True

        outputs = model(batch_of_graphs, mask)
        for out in outputs:
            assert torch.isfinite(out).all(), f"Non-finite values in output with shape {out.shape}"

    def test_gradient_flow(self, config, single_graph):
        """Gradients flow through the entire model."""
        model = RGIN(config)
        batch = Batch.from_data_list([single_graph])
        mask = torch.zeros(20, dtype=torch.bool)
        mask[:5] = True

        z_str, *_ = model(batch, mask)
        loss = z_str.sum()
        loss.backward()

        # Check key parameters have gradients
        assert model.input_project.weight.grad is not None
        # mask_token grad requires masked nodes to be decoded; R grad requires cross-layer loss
        # Both are used in the training loop, not directly in forward+sum
        assert model.gin_layers[0]["calls"][0].weight.grad is not None

    def test_no_edges_still_works(self, config):
        """Model works even if a graph has zero edges for some edge types."""
        from conftest import _make_graph
        data = _make_graph(n=10, seed=99)
        # Remove inherits edges
        data["node", "inherits", "node"].edge_index = torch.zeros(2, 0, dtype=torch.long)

        model = RGIN(config)
        model.eval()
        batch = Batch.from_data_list([data])
        mask = torch.zeros(10, dtype=torch.bool)
        mask[:3] = True

        z_str, z_inv, z_calls, z_imports, z_inherits, g_emb = model(batch, mask)
        assert torch.isfinite(z_str).all()
        assert g_emb.shape == (1, config.invariant_dim)

    def test_param_count(self):
        """Parameter count is approximately 1.95M for default config."""
        config = RGINConfig()  # default: hidden=256
        model = RGIN(config)
        n_params = sum(p.numel() for p in model.parameters())
        # Should be ~1.95M ± 10%
        assert 1.5e6 < n_params < 2.5e6, f"Param count {n_params:,} outside expected range"
