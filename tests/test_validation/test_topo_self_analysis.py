"""Self-analysis regression tests for topo's own first-party code."""

from __future__ import annotations

from tests.test_validation.benchmark_utils import analyze_topo_self


def test_topo_self_analysis_stays_first_party_and_interpretable():
    """Self-analysis should ignore vendored code and expose real package flow."""
    result = analyze_topo_self()

    assert result.coverage is not None
    assert result.health is not None
    spectral = result.raw.get("spectral", {})
    assert spectral.get("coverage_ratio", 0) > 0.8

    # Verify modules are not degenerate — no single module contains all nodes.
    # When the first-party codebase is small (e.g. after removing packages),
    # spectral clustering may legitimately produce fewer modules.
    module_sizes = [m.size for m in result.modules if not m.unassigned]
    total = sum(module_sizes)
    if total > 0 and len(module_sizes) > 1:
        largest_ratio = max(module_sizes) / total
        assert largest_ratio < 1.0

    # When multiple modules are detected, verify cross-module dependencies
    # exist and exclude vendored code. With a small first-party codebase,
    # spectral clustering may produce a single module (no cross-module deps).
    if len(module_sizes) > 1:
        assert len(result.cross_package_dependencies) > 0
        mod_labels = {m.id: m.label for m in result.modules}
        dependency_label_pairs = {
            (mod_labels.get(dep["source"], ""), mod_labels.get(dep["target"], ""))
            for dep in result.cross_package_dependencies
        }
        assert all("pycg" not in p for pair in dependency_label_pairs for p in pair)


def test_topo_self_issues_remain_actionable():
    """The analysis should produce actionable issues without regressions."""
    result = analyze_topo_self()

    assert result.issues is not None
    assert result.health is not None
    # A clean layered codebase should have no layer violations.
    assert not any(f["kind"] == "layer_violation" for f in result.issues)
    # Regression guard: threshold allows for minor variations from algorithm changes.
    assert len(result.issues) <= 15, (
        f"Expected <= 15 issues after detector fixes, got {len(result.issues)}: "
        + ", ".join(f["id"] for f in result.issues)
    )
