"""Tests for SignNet module."""

import torch
import pytest

from topo_model.signnet import SignNet


class TestSignNet:
    def test_output_shape(self):
        net = SignNet(k=16, hidden=64, out_per_eig=2)
        vecs = torch.randn(50, 16)
        vals = torch.rand(50, 16)
        out = net(vecs, vals)
        assert out.shape == (50, 32)  # 16 * 2

    def test_sign_invariance(self):
        """Core property: SignNet(v, λ) == SignNet(-v, λ)."""
        net = SignNet(k=16, hidden=64, out_per_eig=2)
        net.eval()

        vecs = torch.randn(30, 16)
        vals = torch.rand(30, 16)

        out_pos = net(vecs, vals)
        out_neg = net(-vecs, vals)

        # Must be exactly equal (not approximately — the architecture guarantees this)
        assert torch.allclose(out_pos, out_neg, atol=1e-6), \
            f"Max diff: {(out_pos - out_neg).abs().max():.2e}"

    def test_zero_padded_inputs(self):
        """Zero-padded eigenvectors (missing eigenvectors) should produce valid output."""
        net = SignNet(k=16, hidden=64, out_per_eig=2)
        net.eval()

        # Simulate: only first 4 eigenvectors are real, rest are zero-padded
        vecs = torch.zeros(20, 16)
        vals = torch.zeros(20, 16)
        vecs[:, :4] = torch.randn(20, 4)
        vals[:, :4] = torch.rand(20, 4)

        out = net(vecs, vals)
        assert out.shape == (20, 32)
        assert torch.isfinite(out).all()

    def test_different_k(self):
        """Works with different numbers of eigenvectors."""
        for k in [4, 8, 16, 32]:
            net = SignNet(k=k, hidden=32, out_per_eig=2)
            out = net(torch.randn(10, k), torch.rand(10, k))
            assert out.shape == (10, k * 2)

    def test_gradient_flows(self):
        """Gradients flow through SignNet."""
        net = SignNet(k=16, hidden=64, out_per_eig=2)
        vecs = torch.randn(10, 16, requires_grad=True)
        vals = torch.rand(10, 16, requires_grad=True)
        out = net(vecs, vals)
        out.sum().backward()
        assert vecs.grad is not None
        assert vals.grad is not None
