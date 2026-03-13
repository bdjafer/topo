"""Extract derived signals from StructuralAnalysis for benchmark evaluation."""

from __future__ import annotations

from topo_analyzer.analysis import StructuralAnalysis
from topo_benchmark.scoring import compute_ari


def partition_labels(result: StructuralAnalysis) -> dict[str, int]:
    """Extract module partition labels for assigned nodes."""
    labels: dict[str, int] = {}
    for module in result.modules:
        if module.unassigned:
            continue
        for node_id in module.node_ids:
            labels[node_id] = module.id
    return labels


def partition_similarity(a: StructuralAnalysis, b: StructuralAnalysis) -> float:
    """ARI between module partitions of two analyses on shared nodes."""
    return compute_ari(partition_labels(a), partition_labels(b))


def largest_module_ratio(result: StructuralAnalysis) -> float:
    """Largest module ratio from health metrics."""
    if result.health is None:
        return 0.0
    return result.health.largest_module_ratio


def module_count(result: StructuralAnalysis) -> int:
    """Number of non-unassigned modules."""
    return len([m for m in result.modules if not m.unassigned])


def max_anomaly_severity(result: StructuralAnalysis, kind: str) -> float:
    """Max severity among anomalies matching the given kind, default 0."""
    matching = [a.severity for a in result.anomalies if a.kind.value == kind]
    return max(matching) if matching else 0.0


def target_role(result: StructuralAnalysis, node_id: str) -> str | None:
    """Predicted structural role for a specific node."""
    for ra in result.roles:
        if ra.node_id == node_id:
            return ra.role.value
    return None


def target_has_spectral_outlier(result: StructuralAnalysis, node_id: str) -> bool:
    """Whether any spectral_outlier anomaly contains the given node."""
    for anomaly in result.anomalies:
        if anomaly.kind.value == "spectral_outlier" and node_id in anomaly.node_ids:
            return True
    return False


def attribution_at_k(
    result: StructuralAnalysis,
    mutated_region: set[str],
    k: int = 3,
) -> bool:
    """Whether any of the top-k anomalies (by severity) overlap the mutated region."""
    sorted_anomalies = sorted(result.anomalies, key=lambda a: -a.severity)
    for anomaly in sorted_anomalies[:k]:
        if set(anomaly.node_ids) & mutated_region:
            return True
    return False


def has_finding(result: StructuralAnalysis, kind: str) -> bool:
    """Whether the analysis produced a finding of the given kind."""
    return any(f.kind == kind for f in result.findings)


def finding_count(result: StructuralAnalysis, kind: str) -> int:
    """Count of findings matching the given kind."""
    return sum(1 for f in result.findings if f.kind == kind)


def cross_package_dep_count(result: StructuralAnalysis) -> int:
    """Total number of unique cross-package dependency directions."""
    return len(result.cross_package_dependencies)


def has_cross_package_dep(result: StructuralAnalysis, source_pkg: str, target_pkg: str) -> bool:
    """Whether a specific cross-package dependency exists."""
    return any(
        cpd.source_package == source_pkg and cpd.target_package == target_pkg
        for cpd in result.cross_package_dependencies
    )
