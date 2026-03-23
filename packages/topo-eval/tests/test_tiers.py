"""Tests for evaluation tiers 1-4."""

import pytest
import torch

from topo_eval.tier1 import tier1_intrinsic_metrics
from topo_eval.tier2 import tier2_phase2_agreement
from topo_eval.tier3 import tier3_perturbation_test
from topo_eval.tier4 import tier4_structural_consistency


class TestTier1:
    def test_returns_expected_keys(self, model, synthetic_dataset, device):
        result = tier1_intrinsic_metrics(model, synthetic_dataset, device, n_trials=1)
        assert "recon_cosine_sim" in result
        assert "crosslayer_auc" in result
        assert "R_asymmetry" in result
        assert "passes" in result
        assert "tier1_pass" in result

    def test_recon_sim_in_range(self, model, synthetic_dataset, device):
        result = tier1_intrinsic_metrics(model, synthetic_dataset, device, n_trials=1)
        assert -1 <= result["recon_cosine_sim"] <= 1

    def test_r_asymmetry_positive(self, model, synthetic_dataset, device):
        result = tier1_intrinsic_metrics(model, synthetic_dataset, device, n_trials=1)
        assert result["R_asymmetry"] >= 0


class TestTier2:
    def test_returns_expected_keys(self, model, synthetic_dataset, device):
        result = tier2_phase2_agreement(model, synthetic_dataset, device)
        assert "rank_correlation_mean" in result
        assert "topk_overlap_mean" in result
        assert "per_repo" in result
        assert "tier2_pass" in result

    def test_correlation_in_range(self, model, synthetic_dataset, device):
        result = tier2_phase2_agreement(model, synthetic_dataset, device)
        assert -1 <= result["rank_correlation_mean"] <= 1

    def test_overlap_in_range(self, model, synthetic_dataset, device):
        result = tier2_phase2_agreement(model, synthetic_dataset, device)
        assert 0 <= result["topk_overlap_mean"] <= 1


class TestTier3:
    def test_returns_expected_keys(self, model, synthetic_dataset, device):
        result = tier3_perturbation_test(model, synthetic_dataset, device, n_trials=2)
        assert "perturbation_sensitivity_mean" in result
        assert "perturbation_specificity_mean" in result
        assert "perturbation_precision_mean" in result
        assert "perturbation_error_delta_mean" in result
        assert "control_sensitivity_mean" in result
        assert "tier3_pass" in result

    def test_sensitivity_in_range(self, model, synthetic_dataset, device):
        result = tier3_perturbation_test(model, synthetic_dataset, device, n_trials=2)
        assert 0 <= result["perturbation_sensitivity_mean"] <= 1

    def test_specificity_in_range(self, model, synthetic_dataset, device):
        result = tier3_perturbation_test(model, synthetic_dataset, device, n_trials=2)
        assert 0 <= result["perturbation_specificity_mean"] <= 1


class TestTier4:
    def test_returns_expected_keys(self, model, synthetic_dataset, device):
        result = tier4_structural_consistency(model, synthetic_dataset, device)
        assert "nmi_mean" in result
        assert "error_degree_corr_mean" in result
        assert "per_repo" in result
        assert "tier4_pass" in result

    def test_nmi_in_range(self, model, synthetic_dataset, device):
        result = tier4_structural_consistency(model, synthetic_dataset, device)
        # NMI lift can be negative (z_inv worse than random baseline)
        assert -1 <= result["nmi_mean"] <= 1

    def test_error_degree_corr_in_range(self, model, synthetic_dataset, device):
        result = tier4_structural_consistency(model, synthetic_dataset, device)
        assert -1 <= result["error_degree_corr_mean"] <= 1
