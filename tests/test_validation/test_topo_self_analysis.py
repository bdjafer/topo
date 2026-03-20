"""Self-analysis regression tests for topo's own first-party code."""

from __future__ import annotations

from topo_formatter.text import format_text
from tests.test_validation.benchmark_utils import analyze_topo_self


def test_topo_self_analysis_stays_first_party_and_interpretable():
    """Self-analysis should ignore vendored code and expose real package flow."""
    result = analyze_topo_self()

    assert result.coverage is not None
    assert result.health is not None
    spectral = result.raw.get("spectral", {})
    assert spectral.get("coverage_ratio", 0) > 0.8

    # Verify modules are not degenerate — no single module contains all nodes.
    module_sizes = [m.size for m in result.modules if not m.unassigned]
    total = sum(module_sizes)
    if total > 0:
        largest_ratio = max(module_sizes) / total
        assert largest_ratio < 1.0

    # Build module ID → label mapping for cross-module dependency verification.
    mod_labels = {m.id: m.label for m in result.modules}
    dependency_label_pairs = {
        (mod_labels.get(dep["source"], ""), mod_labels.get(dep["target"], ""))
        for dep in result.cross_package_dependencies
    }
    assert ("topo_cli", "topo_parser") in dependency_label_pairs
    assert all("pycg" not in p for pair in dependency_label_pairs for p in pair)


def test_topo_self_summary_and_findings_remain_actionable():
    """The default summary should stay findings-first and package-oriented."""
    result = analyze_topo_self()
    data = result.raw
    summary = format_text(data)

    assert "Issues" in summary or "issues" in summary.lower()
    assert "Architecture" in summary or "architecture" in summary.lower()
    assert "Health" in summary or "health" in summary.lower()
    # A clean layered codebase should have no reverse dependency findings.
    assert not any(f["kind"] == "reverse_dependency" for f in result.findings)
    # Regression guard: threshold allows for minor variations from algorithm changes.
    assert len(result.findings) <= 15, (
        f"Expected <= 15 findings after detector fixes, got {len(result.findings)}: "
        + ", ".join(f["id"] for f in result.findings)
    )
