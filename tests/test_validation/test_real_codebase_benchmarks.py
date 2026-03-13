"""Hermetic fixture benchmarks that stand in for real architectural patterns."""

from __future__ import annotations

from tests.test_validation.benchmark_utils import (
    analyze_fixture,
    compute_nmi_from_mappings,
    labels_by_top_package,
    labels_from_modules,
)


def test_layered_app_aligns_with_package_structure():
    """A layered fixture should cluster close to the obvious package baseline."""
    result = analyze_fixture("layered_app", n_modules=3)
    predicted = labels_from_modules(result)
    baseline = labels_by_top_package(list(predicted))
    nmi = compute_nmi_from_mappings(predicted, baseline)

    assert result.module_detection.silhouette is not None
    assert nmi > 0.6
    assert not any(
        finding.kind == "reverse_dependency"
        for finding in result.findings
    )


def test_reverse_flow_fixture_reports_bidirectional_dependency():
    """A fixture with reverse package flow should surface a reverse dependency finding."""
    result = analyze_fixture("reverse_flow_app")

    dependency_pairs = {
        (dependency.source_package, dependency.target_package)
        for dependency in result.cross_package_dependencies
    }
    assert ("api", "data") in dependency_pairs
    assert ("data", "api") in dependency_pairs
    assert any(
        finding.kind == "reverse_dependency" and "api and data" in finding.title
        for finding in result.findings
    )
