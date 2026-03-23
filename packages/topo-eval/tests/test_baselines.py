"""Tests for ablation baselines."""

import pytest

from topo_eval.baselines import run_baselines


class TestBaselines:
    def test_all_baselines_run(self, synthetic_dataset):
        results = run_baselines(synthetic_dataset, n_trials=2, seed=42)
        assert "random" in results
        assert "phase2_local_variation" in results
        assert "centroid_distance" in results
        assert "degree_only" in results

    def test_baseline_result_keys(self, synthetic_dataset):
        results = run_baselines(synthetic_dataset, n_trials=2, seed=42)
        for name, res in results.items():
            assert "sensitivity_mean" in res, f"{name} missing sensitivity_mean"
            assert "specificity_mean" in res, f"{name} missing specificity_mean"
            assert "precision_mean" in res, f"{name} missing precision_mean"
            assert "error_delta_mean" in res, f"{name} missing error_delta_mean"
            assert "n_trials" in res, f"{name} missing n_trials"

    def test_random_baseline_near_chance(self, synthetic_dataset):
        results = run_baselines(synthetic_dataset, n_trials=3, seed=42)
        random_sens = results["random"]["sensitivity_mean"]
        # Random should be near 0.2 (20% threshold), allow wide tolerance
        assert 0.0 <= random_sens <= 0.6, f"Random sensitivity {random_sens} unexpectedly extreme"

    def test_values_in_range(self, synthetic_dataset):
        results = run_baselines(synthetic_dataset, n_trials=2, seed=42)
        for name, res in results.items():
            assert 0 <= res["sensitivity_mean"] <= 1, f"{name} sensitivity out of range"
            assert 0 <= res["specificity_mean"] <= 1, f"{name} specificity out of range"
            assert 0 <= res["precision_mean"] <= 1, f"{name} precision out of range"
