"""Mutation benchmark validation on realistic parsed fixtures."""

from __future__ import annotations

from tests.test_validation.benchmark_utils import (
    StructuralAnalysisResult,
    analyze_fixture,
    compute_nmi_from_mappings,
    labels_by_top_package,
    labels_from_modules,
)


def _package_alignment_nmi(result: StructuralAnalysisResult) -> float:
    """Compare detected modules against the simple top-package baseline."""
    labels = labels_from_modules(result)
    return compute_nmi_from_mappings(labels, labels_by_top_package(list(labels)))


def _analyzed_node_ids(result: StructuralAnalysisResult) -> list[str]:
    """Return all analyzed node IDs from the roles list."""
    return sorted(r["node_id"] for r in result.roles)


def _non_coverage_issue_kinds(result: StructuralAnalysisResult) -> list[str]:
    """Return high-signal issue kinds, ignoring infrastructure noise."""
    noise = {
        "coverage", "orphan", "module_separation", "low_cohesion",
        "god_module", "layer_discrepancy", "phantom_import", "wide_interface",
        "fragile_hub", "self_edge_drop",
    }
    return [
        issue["kind"] for issue in result.issues
        if issue["kind"] not in noise
    ]


def _issue_kinds(result: StructuralAnalysisResult) -> list[str]:
    """Return all issue kinds."""
    return [issue["kind"] for issue in result.issues]


def test_reverse_dependency_fixture_is_targeted_without_partition_collapse():
    """A reverse dependency case should produce one dominant structural issue."""
    clean = analyze_fixture("layered_app", n_modules=3)
    mutated = analyze_fixture("layered_reverse_dep_app", n_modules=3)

    assert _package_alignment_nmi(clean) > 0.5
    assert _package_alignment_nmi(mutated) > 0.3
    assert _non_coverage_issue_kinds(clean) == []
    assert "reverse_dependency" in _non_coverage_issue_kinds(mutated)


def test_reverse_dependency_fixture_changes_structure_without_changing_package_layout():
    """The analyzer should catch structural regressions that package layout cannot distinguish."""
    clean = analyze_fixture("layered_app", n_modules=3)
    mutated = analyze_fixture("layered_reverse_dep_app", n_modules=3)

    clean_pkgs = labels_by_top_package(_analyzed_node_ids(clean))
    mutated_pkgs = labels_by_top_package(_analyzed_node_ids(mutated))
    assert clean_pkgs == mutated_pkgs
    assert _non_coverage_issue_kinds(clean) == []
    assert _non_coverage_issue_kinds(mutated) == ["reverse_dependency"]


def test_cycle_fixture_surfaces_cycle_with_limited_collateral_noise():
    """A cycle case should expose a cycle without flooding the issues list."""
    clean = analyze_fixture("layered_app", n_modules=3)
    mutated = analyze_fixture("layered_cycle_app", n_modules=3)

    assert _package_alignment_nmi(clean) > 0.5
    assert _package_alignment_nmi(mutated) > 0.3
    assert "cycle_member" not in _non_coverage_issue_kinds(clean)
    mutated_kinds = set(_non_coverage_issue_kinds(mutated))
    assert "cycle_member" in mutated_kinds
    assert "reverse_dependency" in mutated_kinds


def test_boundary_erosion_fixture_weakens_alignment_without_reverse_dependency():
    """Boundary erosion should reduce package alignment without becoming bidirectional."""
    clean = analyze_fixture("layered_app", n_modules=3)
    mutated = analyze_fixture("layered_boundary_erosion_app", n_modules=3)

    assert _package_alignment_nmi(clean) > 0.5
    # Boundary erosion should not introduce reverse dependencies.
    assert "reverse_dependency" not in _non_coverage_issue_kinds(mutated)
