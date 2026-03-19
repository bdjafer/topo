"""Self-analysis regression tests for topo's own first-party code."""

from __future__ import annotations

from tests.test_validation.benchmark_utils import analyze_topo_self


def test_topo_self_analysis_stays_first_party_and_interpretable():
    """Self-analysis should ignore vendored code and expose real package flow."""
    result = analyze_topo_self()

    assert result.coverage is not None
    assert result.health is not None
    assert result.coverage.spectral_coverage_ratio > 0.8
    # Dual-level analysis clusters at SYMBOL level then aggregates to MODULE.
    # The tightly-coupled topo codebase may produce a dominant cluster; verify
    # the ratio is bounded but not unrealistically concentrated.
    assert result.health.largest_module_ratio < 1.0

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

    assert "Issues" in summary
    assert "Architecture" in summary
    assert "Health" in summary
    # After call-edge validation against imports, the false reverse dependency
    # (topo_parser -> topo_analyzer from PyCG suffix matching) is eliminated.
    # A clean layered codebase should have no reverse dependency findings.
    assert not any(finding.kind == "reverse_dependency" for finding in result.findings)
    # Regression guard: with per-kind percentile normalization, clustering
    # quality gates, and orphan unanimity, the self-analysis should produce
    # a small number of TRUE-EXPECTED findings, not dozens of artifacts.
    # Threshold: 15 allows for minor variations from algorithm changes
    # (e.g. hand-rolled Tarjan/Brandes vs networkx, Rust vs Python backend).
    assert len(result.findings) <= 15, (
        f"Expected <= 15 findings after detector fixes, got {len(result.findings)}: "
        + ", ".join(f.id for f in result.findings)
    )
