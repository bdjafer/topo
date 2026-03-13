"""Architecture recovery dimension scorer."""

from __future__ import annotations

from topo_analyzer.analysis import StructuralAnalysis
from topo_parser.graph import CodeGraph

from topo_benchmark.scoring import compute_ari, compute_boundary_f1, compute_coverage, geometric_mean
from topo_benchmark.signals import partition_labels


def score_architecture_case(
    graph: CodeGraph,
    labels: dict,
    analysis: StructuralAnalysis,
    baseline_partitions: dict[str, dict[str, object]] | None = None,
) -> dict:
    """Score a single architecture recovery case.

    Returns dict with ARI, BoundaryF1, Coverage, dimension score, and guardrails.
    """
    gold_nodes: dict[str, object] = labels["included_nodes"]
    excluded = set(labels.get("excluded_nodes", []))

    # Predicted partition (module labels from analysis)
    predicted = {k: v for k, v in partition_labels(analysis).items() if k not in excluded}

    # Only evaluate on included gold nodes
    gold_filtered = {k: v for k, v in gold_nodes.items() if k not in excluded}

    # ARI
    ari = max(0.0, compute_ari(predicted, gold_filtered))

    # Boundary F1
    edges = [
        (e.source, e.target)
        for e in analysis.graph.edges
        if e.source in gold_filtered and e.target in gold_filtered
    ]
    boundary_f1 = compute_boundary_f1(edges, predicted, gold_filtered)

    # Coverage
    coverage = compute_coverage(predicted, set(gold_filtered.keys()))

    # Dimension score
    score = geometric_mean(ari, boundary_f1) * coverage

    # Guardrails
    coverage_floor = 0.5
    coverage_ok = coverage >= coverage_floor

    baseline_ok = True
    baseline_scores: dict[str, float] = {}
    if baseline_partitions:
        for name, bp in baseline_partitions.items():
            bp_filtered = {k: v for k, v in bp.items() if k in gold_filtered}
            b_ari = max(0.0, compute_ari(bp_filtered, gold_filtered))
            b_f1 = compute_boundary_f1(edges, bp_filtered, gold_filtered)
            b_cov = compute_coverage(bp_filtered, set(gold_filtered.keys()))
            b_score = geometric_mean(b_ari, b_f1) * b_cov
            baseline_scores[name] = b_score
            if score < b_score:
                baseline_ok = False

    return {
        "ari": ari,
        "boundary_f1": boundary_f1,
        "coverage": coverage,
        "score": score,
        "guardrails": {
            "coverage_ok": coverage_ok,
            "baseline_ok": baseline_ok,
        },
        "baseline_scores": baseline_scores,
    }
