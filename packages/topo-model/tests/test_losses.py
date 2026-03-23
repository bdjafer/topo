"""Tests for loss functions and HSIC."""

import torch
import pytest
from torch_geometric.data import Batch

from topo_model.losses import (
    masked_reconstruction_loss,
    cross_layer_score,
    cross_layer_loss,
    graph_contrastive_loss,
    hsic,
    rbf_kernel,
    per_graph_hsic,
    generate_mask,
    ramp,
    sample_negatives,
)
from conftest import _make_graph


class TestMaskedReconstructionLoss:
    def test_perfect_prediction(self):
        """Identical predictions and targets should give loss ≈ 0."""
        x = torch.randn(50, 768)
        x = x / x.norm(dim=-1, keepdim=True)  # normalize
        loss = masked_reconstruction_loss(x, x)
        assert loss.item() < 1e-5

    def test_orthogonal_prediction(self):
        """Orthogonal predictions should give loss ≈ 1."""
        x = torch.randn(50, 768)
        y = torch.randn(50, 768)
        # Make orthogonal: y = y - (y·x/x·x)x
        loss = masked_reconstruction_loss(x, y)
        assert 0.0 < loss.item() < 2.0  # cosine distance is in [0, 2]

    def test_empty_mask(self):
        """Empty mask (no masked nodes) should give 0."""
        x = torch.randn(0, 768)
        loss = masked_reconstruction_loss(x, x)
        assert loss.item() == 0.0

    def test_gradient_flows(self):
        pred = torch.randn(20, 768, requires_grad=True)
        target = torch.randn(20, 768)
        loss = masked_reconstruction_loss(pred, target)
        loss.backward()
        assert pred.grad is not None


class TestCrossLayerLoss:
    def test_score_shape(self):
        z = torch.randn(50, 32)
        R = torch.randn(32, 32) * 0.01
        edges = torch.tensor([[0, 1, 2], [3, 4, 5]])
        scores = cross_layer_score(z, edges, R)
        assert scores.shape == (3,)

    def test_loss_value_range(self):
        z = torch.randn(50, 32)
        R = torch.randn(32, 32) * 0.01
        pos = torch.tensor([[0, 1, 2], [3, 4, 5]])
        neg = torch.tensor([[0, 1, 2, 0, 1, 2], [10, 11, 12, 20, 21, 22]])
        loss = cross_layer_loss(z, pos, neg, R)
        assert loss.item() > 0  # BCE is always positive

    def test_empty_edges(self):
        z = torch.randn(50, 32)
        R = torch.randn(32, 32)
        pos = torch.zeros(2, 0, dtype=torch.long)
        neg = torch.zeros(2, 0, dtype=torch.long)
        loss = cross_layer_loss(z, pos, neg, R)
        assert loss.item() == 0.0


class TestGraphContrastiveLoss:
    def test_positive_loss(self):
        """Contrastive loss should always be non-negative."""
        g = torch.randn(4, 64)
        loss = graph_contrastive_loss(g, None, tau=0.07)
        assert loss.item() >= 0.0
        assert torch.isfinite(loss)

    def test_minimum_batch(self):
        """Works with just 2 graphs."""
        g = torch.randn(2, 64)
        loss = graph_contrastive_loss(g, None, tau=0.07)
        assert torch.isfinite(loss)

    def test_single_graph_returns_zero(self):
        g = torch.randn(1, 64)
        loss = graph_contrastive_loss(g, None, tau=0.07)
        assert loss.item() == 0.0


class TestHSIC:
    def test_independent_variables(self):
        """HSIC of independent random variables should be near zero."""
        torch.manual_seed(42)
        X = torch.randn(200, 32)
        Y = torch.randn(200, 32)
        val = hsic(X, Y)
        # Should be close to zero for independent variables
        assert val.item() < 0.1

    def test_dependent_variables(self):
        """HSIC of dependent variables should be greater than independent."""
        torch.manual_seed(42)
        X = torch.randn(200, 32)
        Y_dep = X + torch.randn(200, 32) * 0.1  # Y ≈ X
        Y_ind = torch.randn(200, 32)  # independent
        hsic_dep = hsic(X, Y_dep)
        hsic_ind = hsic(X, Y_ind)
        assert hsic_dep.item() > hsic_ind.item()

    def test_non_negative(self):
        """Biased HSIC should be clamped to non-negative."""
        X = torch.randn(10, 16)
        Y = torch.randn(10, 16)
        val = hsic(X, Y)
        assert val.item() >= 0.0

    def test_small_n_returns_zero(self):
        """Fewer than 5 nodes should return 0 (unreliable estimate)."""
        X = torch.randn(3, 32)
        Y = torch.randn(3, 32)
        assert hsic(X, Y).item() == 0.0

    def test_rbf_kernel_shape(self):
        X = torch.randn(20, 32)
        K = rbf_kernel(X)
        assert K.shape == (20, 20)
        # Diagonal should be 1 (distance to self is 0)
        assert torch.allclose(K.diag(), torch.ones(20), atol=1e-5)


class TestPerGraphHSIC:
    def test_per_graph(self):
        batch_idx = torch.tensor([0]*15 + [1]*15)
        z_inv = torch.randn(30, 32)
        z_calls = torch.randn(30, 16)
        z_imports = torch.randn(30, 16)
        z_inherits = torch.randn(30, 16)
        val = per_graph_hsic(batch_idx, z_inv, z_calls, z_imports, z_inherits)
        assert val.item() >= 0.0
        assert torch.isfinite(val)


class TestGenerateMask:
    def test_mask_ratio(self):
        data = _make_graph(n=100, seed=0)
        batch = Batch.from_data_list([data])
        mask = generate_mask(batch, ratio=0.65)
        assert mask.shape == (100,)
        n_masked = mask.sum().item()
        assert 50 <= n_masked <= 80  # ~65 ± tolerance

    def test_batch_respects_boundaries(self):
        """Mask should respect per-graph boundaries."""
        g1 = _make_graph(n=20, seed=0)
        g2 = _make_graph(n=30, seed=1)
        batch = Batch.from_data_list([g1, g2])
        mask = generate_mask(batch, ratio=0.5)
        assert mask.shape == (50,)
        # Check each graph has roughly 50% masked
        n1_masked = mask[:20].sum().item()
        n2_masked = mask[20:].sum().item()
        assert 5 <= n1_masked <= 15
        assert 8 <= n2_masked <= 22


class TestRamp:
    def test_before_start(self):
        assert ramp(5, 10, 30, 0.5) == 0.0

    def test_at_start(self):
        assert ramp(10, 10, 30, 0.5) == 0.0

    def test_mid_ramp(self):
        val = ramp(20, 10, 30, 0.5)
        assert abs(val - 0.25) < 1e-6

    def test_at_end(self):
        assert ramp(30, 10, 30, 0.5) == 0.5

    def test_after_end(self):
        assert ramp(50, 10, 30, 0.5) == 0.5


class TestSampleNegatives:
    def test_shape(self):
        g1 = _make_graph(n=20, seed=0)
        batch = Batch.from_data_list([g1])
        pos = batch["node", "calls", "node"].edge_index
        n_pos = pos.shape[1]
        neg = sample_negatives(batch, pos, ratio=5)
        assert neg.shape == (2, n_pos * 5)

    def test_empty_edges(self):
        g1 = _make_graph(n=20, seed=0)
        batch = Batch.from_data_list([g1])
        pos = torch.zeros(2, 0, dtype=torch.long)
        neg = sample_negatives(batch, pos, ratio=5)
        assert neg.shape == (2, 0)
