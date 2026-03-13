"""Self-analysis regression tests for topo's own first-party code."""

from __future__ import annotations

from tests.test_validation.benchmark_utils import analyze_topo_self


def test_topo_self_analysis_stays_first_party_and_interpretable():
    """Self-analysis should ignore vendored code and expose real package flow."""
    result = analyze_topo_self()

    assert result.coverage is not None
    assert result.health is not None
    assert result.coverage.spectral_coverage_ratio > 0.8
    assert result.health.largest_module_ratio < 0.5

    dependency_pairs = {
        (dependency.source_package, dependency.target_package)
        for dependency in result.cross_package_dependencies
    }
    assert ("topo_cli", "topo_parser") in dependency_pairs
    assert ("topo_cli", "topo_analyzer") in dependency_pairs
    assert ("topo_analyzer", "topo_parser") in dependency_pairs
    assert all("pycg" not in pair for dependency in dependency_pairs for pair in dependency)


def test_topo_self_summary_and_findings_remain_actionable():
    """The default summary should stay findings-first and package-oriented."""
    result = analyze_topo_self()
    summary = result.summary()

    assert "Findings:" in summary
    assert "Package flow:" in summary
    assert "Scope roots:" in summary
    assert len(result.findings) <= 7
    assert any(finding.kind == "reverse_dependency" for finding in result.findings)
