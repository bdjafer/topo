"""Mutation benchmark validation on realistic parsed fixtures."""

from __future__ import annotations

from topo_analyzer.analysis import StructuralAnalysis

from tests.test_validation.benchmark_utils import (
    analyze_fixture,
    compute_nmi_from_mappings,
    labels_by_top_package,
    labels_from_modules,
)


def _package_alignment_nmi(result: StructuralAnalysis) -> float:
    """Compare detected modules against the simple top-package baseline."""
    labels = labels_from_modules(result)
    return compute_nmi_from_mappings(labels, labels_by_top_package(list(labels)))


def _package_partition(result: StructuralAnalysis) -> dict[str, str]:
    """Return the naive package partition for all analyzed nodes."""
    return labels_by_top_package(sorted(result.graph.nodes))


def _non_coverage_finding_kinds(result: StructuralAnalysis) -> list[str]:
    """Return high-signal finding kinds, ignoring infrastructure noise.

    These finding types trigger on small synthetic fixtures (3 modules,
    ~6 nodes each) where cluster sizes, degree distributions, and
    cohesion metrics don't reflect real architectural problems. The
    mutation benchmarks focus on structural signals — reverse_dependency,
    cycle_member — that are meaningful even at this scale.
    """
    noise = {
        "coverage", "orphan", "module_separation", "low_cohesion",
        "god_module", "layer_discrepancy", "phantom_import", "wide_interface",
    }
    return [
        finding.kind for finding in result.findings
        if finding.kind not in noise
    ]


def _anomaly_kinds(result: StructuralAnalysis) -> list[str]:
    """Return anomaly kinds in priority order."""
    return [anomaly.kind.value for anomaly in result.anomalies]


def test_reverse_dependency_fixture_is_targeted_without_partition_collapse():
    """A reverse dependency case should produce one dominant structural finding."""
    clean = analyze_fixture("layered_app", n_modules=3)
    mutated = analyze_fixture("layered_reverse_dep_app", n_modules=3)

    assert _package_alignment_nmi(clean) > 0.7
    assert _package_alignment_nmi(mutated) > 0.7
    assert _non_coverage_finding_kinds(clean) == []
    assert "reverse_dependency" in _non_coverage_finding_kinds(mutated)
    # Symbol-level analysis correctly surfaces cross-module anomalies for
    # the reverse dependency — these are true positives from richer analysis.
    mutated_anomaly_kinds = set(_anomaly_kinds(mutated))
    assert mutated_anomaly_kinds <= {"cross_module", "layer_discrepancy"}


def test_reverse_dependency_fixture_changes_structure_without_changing_package_layout():
    """The analyzer should catch structural regressions that package layout cannot distinguish."""
    clean = analyze_fixture("layered_app", n_modules=3)
    mutated = analyze_fixture("layered_reverse_dep_app", n_modules=3)

    assert _package_partition(clean) == _package_partition(mutated)
    assert _non_coverage_finding_kinds(clean) == []
    assert _non_coverage_finding_kinds(mutated) == ["reverse_dependency"]


def test_cycle_fixture_surfaces_cycle_with_limited_collateral_noise():
    """A cycle case should expose a cycle without flooding the findings list."""
    clean = analyze_fixture("layered_app", n_modules=3)
    mutated = analyze_fixture("layered_cycle_app", n_modules=3)

    assert _package_alignment_nmi(clean) > 0.7
    assert _package_alignment_nmi(mutated) > 0.7
    assert "cycle_member" not in _non_coverage_finding_kinds(clean)
    mutated_kinds = set(_non_coverage_finding_kinds(mutated))
    assert "cycle_member" in mutated_kinds
    assert "reverse_dependency" in mutated_kinds
    mutated_anomaly_kinds = set(_anomaly_kinds(mutated))
    assert "cycle_member" in mutated_anomaly_kinds


def test_boundary_erosion_fixture_weakens_alignment_without_reverse_dependency():
    """Boundary erosion should reduce package alignment without becoming bidirectional."""
    clean = analyze_fixture("layered_app", n_modules=3)
    mutated = analyze_fixture("layered_boundary_erosion_app", n_modules=3)

    assert _package_alignment_nmi(clean) > 0.7
    # Symbol-level analysis shows stronger module collapse for eroded boundaries.
    assert _package_alignment_nmi(mutated) < _package_alignment_nmi(clean)
    assert mutated.health is not None and clean.health is not None
    assert mutated.health.largest_module_ratio > clean.health.largest_module_ratio
    assert "reverse_dependency" not in _non_coverage_finding_kinds(mutated)
    assert "cross_module" in _anomaly_kinds(mutated)
