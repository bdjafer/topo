"""Baseline comparisons for structural clustering quality."""

from __future__ import annotations

from tests.test_validation.benchmark_utils import (
    analyze_fixture,
    compute_nmi_from_mappings,
    labels_by_top_package,
    labels_from_modules,
)


def test_layered_app_beats_trivial_single_cluster_baseline():
    """Structural clustering should beat a trivial one-cluster baseline."""
    result = analyze_fixture("layered_app")
    predicted = labels_from_modules(result)
    package_labels = labels_by_top_package(list(predicted))
    one_cluster = {node_id: 0 for node_id in predicted}

    structural_nmi = compute_nmi_from_mappings(predicted, package_labels)
    trivial_nmi = compute_nmi_from_mappings(one_cluster, package_labels)

    assert structural_nmi > trivial_nmi
