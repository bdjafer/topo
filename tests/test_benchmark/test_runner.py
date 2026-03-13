"""Integration test for the benchmark runner."""

from __future__ import annotations

from pathlib import Path

from topo_benchmark.runner import run_benchmark


def _dataset_root() -> Path:
    return Path(__file__).resolve().parents[2] / "benchmark" / "datasets"


def test_benchmark_run_produces_valid_scorecard(tmp_path: Path):
    """Full benchmark run should produce a valid scorecard with all dimensions."""
    scorecard = run_benchmark(
        tier="analyzer",
        split="public",
        dataset_root=_dataset_root(),
        output_dir=tmp_path,
    )

    # Scorecard structure
    assert scorecard.tier == "analyzer"
    assert scorecard.split == "public"
    assert 0 <= scorecard.overall_primary <= 1

    # All 4 dimensions present
    assert "architecture_recovery" in scorecard.dimensions
    assert "mutation_ranking" in scorecard.dimensions
    assert "stability" in scorecard.dimensions
    assert "anomaly_precision_calibration" in scorecard.dimensions

    for score in scorecard.dimensions.values():
        assert 0 <= score <= 1

    # Guardrails are booleans
    for v in scorecard.guardrails.values():
        assert isinstance(v, bool)

    assert scorecard.promotion_decision in ("pass", "fail")

    # Output artifacts exist
    assert (tmp_path / "scorecard.json").exists()
    assert (tmp_path / "dimensions.json").exists()
    assert (tmp_path / "per_case.jsonl").exists()
    assert (tmp_path / "failures.json").exists()
    assert (tmp_path / "summary.md").exists()


def test_benchmark_mutation_ranking_above_zero(tmp_path: Path):
    """Mutation ranking should produce non-zero score with correct expectations."""
    scorecard = run_benchmark(
        tier="analyzer",
        split="public",
        dataset_root=_dataset_root(),
        output_dir=tmp_path,
    )
    assert scorecard.dimensions["mutation_ranking"] > 0.5


def test_benchmark_no_failures(tmp_path: Path):
    """No cases should fail with errors."""
    import json

    run_benchmark(
        tier="analyzer",
        split="public",
        dataset_root=_dataset_root(),
        output_dir=tmp_path,
    )
    failures = json.loads((tmp_path / "failures.json").read_text())
    assert failures == []
