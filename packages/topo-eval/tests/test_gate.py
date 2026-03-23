"""Tests for go/no-go gate logic."""

import pytest

from topo_eval.gate import go_no_go_gate


def _make_passing_results():
    """Create results that should pass all gate checks."""
    tier1 = {
        "tier1_pass": True,
        "recon_cosine_sim": 0.8,
        "crosslayer_auc": 0.85,
        "R_asymmetry": 0.5,
        "passes": {"recon_cosine_sim": True, "crosslayer_auc": True, "R_asymmetry": True},
    }
    tier2 = {
        "tier2_pass": True,
        "rank_correlation_mean": 0.5,
        "topk_overlap_mean": 0.4,
        "passes": {"rank_correlation_mean": True, "topk_overlap_mean": True},
    }
    tier3 = {
        "tier3_pass": True,
        "perturbation_sensitivity_mean": 0.7,
        "perturbation_specificity_mean": 0.8,
        "perturbation_precision_mean": 0.5,
        "perturbation_error_delta_mean": 0.1,
        "control_sensitivity_mean": 0.20,
    }
    tier4 = {
        "tier4_pass": True,
        "nmi_mean": 0.5,
        "error_degree_corr_mean": 0.2,
    }
    baselines = {
        "random": {"precision_mean": 0.1, "sensitivity_mean": 0.2},
        "phase2_local_variation": {"precision_mean": 0.3, "sensitivity_mean": 0.3},
        "centroid_distance": {"precision_mean": 0.25, "sensitivity_mean": 0.25},
        "degree_only": {"precision_mean": 0.15, "sensitivity_mean": 0.2},
    }
    return tier1, tier2, tier3, tier4, baselines


class TestGoNoGoGate:
    def test_all_pass(self):
        tier1, tier2, tier3, tier4, baselines = _make_passing_results()
        result = go_no_go_gate(tier1, tier2, tier3, tier4, baselines)
        assert result["gate_pass"] is True
        assert "DEPLOY" in result["recommendation"]
        assert len(result["diagnostics"]) == 0

    def test_tier1_failure(self):
        tier1, tier2, tier3, tier4, baselines = _make_passing_results()
        tier1["tier1_pass"] = False
        result = go_no_go_gate(tier1, tier2, tier3, tier4, baselines)
        assert result["gate_pass"] is False
        assert any("Tier 1" in d for d in result["diagnostics"])

    def test_tier2_failure(self):
        tier1, tier2, tier3, tier4, baselines = _make_passing_results()
        tier2["tier2_pass"] = False
        result = go_no_go_gate(tier1, tier2, tier3, tier4, baselines)
        assert result["gate_pass"] is False
        assert any("Tier 2" in d for d in result["diagnostics"])

    def test_sensitivity_failure(self):
        tier1, tier2, tier3, tier4, baselines = _make_passing_results()
        tier3["perturbation_sensitivity_mean"] = 0.3  # Below 0.6 threshold
        result = go_no_go_gate(tier1, tier2, tier3, tier4, baselines)
        assert result["gate_pass"] is False
        assert result["checks"]["tier3_sensitivity"] is False

    def test_precision_improvement_failure(self):
        tier1, tier2, tier3, tier4, baselines = _make_passing_results()
        # Phase 3 precision same as Phase 2 — no improvement
        tier3["perturbation_precision_mean"] = 0.3
        baselines["phase2_local_variation"]["precision_mean"] = 0.3
        result = go_no_go_gate(tier1, tier2, tier3, tier4, baselines)
        assert result["gate_pass"] is False
        assert result["checks"]["precision_improvement"] is False

    def test_baseline_comparison(self):
        tier1, tier2, tier3, tier4, baselines = _make_passing_results()
        # Phase 3 precision lower than one baseline
        baselines["centroid_distance"]["precision_mean"] = 0.6
        tier3["perturbation_precision_mean"] = 0.5
        result = go_no_go_gate(tier1, tier2, tier3, tier4, baselines)
        assert result["gate_pass"] is False
        assert result["checks"]["beats_all_baselines"] is False

    def test_control_sanity_failure(self):
        tier1, tier2, tier3, tier4, baselines = _make_passing_results()
        tier3["control_sensitivity_mean"] = 0.5  # Way too high
        result = go_no_go_gate(tier1, tier2, tier3, tier4, baselines)
        assert result["gate_pass"] is False
        assert result["checks"]["control_sanity"] is False

    def test_diagnostics_list_all_failures(self):
        tier1, tier2, tier3, tier4, baselines = _make_passing_results()
        tier1["tier1_pass"] = False
        tier2["tier2_pass"] = False
        tier3["perturbation_sensitivity_mean"] = 0.3
        result = go_no_go_gate(tier1, tier2, tier3, tier4, baselines)
        assert len(result["diagnostics"]) >= 3
