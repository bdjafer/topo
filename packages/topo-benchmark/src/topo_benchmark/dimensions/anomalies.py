"""Anomaly precision and calibration dimension scorer."""

from __future__ import annotations

from topo_analyzer.analysis import StructuralAnalysis

from topo_benchmark.scoring import (
    compute_average_precision,
    compute_brier,
    compute_ece,
    compute_precision_at_k,
    geometric_mean,
)


def _match_anomaly(predicted_nodes: set[str], gold_nodes: set[str], iou_threshold: float = 0.5) -> bool:
    """Check if predicted anomaly matches gold by IoU on node sets.

    The IoU threshold of 0.5 follows the standard object-detection convention
    (PASCAL VOC) adapted to node-set overlap.  At 0.5 the predicted region must
    share at least half its nodes (by union) with the gold region to count as a
    true positive.  This balances tolerance for partial overlap against
    requiring meaningful agreement.
    """
    if not predicted_nodes or not gold_nodes:
        return False
    intersection = predicted_nodes & gold_nodes
    union = predicted_nodes | gold_nodes
    return len(intersection) / len(union) >= iou_threshold


def score_anomaly_case(
    case_id: str,
    analysis: StructuralAnalysis,
    gold: dict,
) -> dict:
    """Score a single anomaly case.

    Gold format:
    {"anomalies": [{"kind": "...", "region_nodes": [...], ...}, ...]}
    """
    gold_anomalies = gold.get("anomalies", [])

    # Build gold region sets
    gold_regions: list[tuple[str | None, set[str]]] = []
    for ga in gold_anomalies:
        kind = ga.get("kind")
        nodes = set(ga.get("region_nodes", []))
        gold_regions.append((kind, nodes))

    # Match predicted anomalies against gold
    predicted = sorted(analysis.anomalies, key=lambda a: -a.severity)

    scores: list[float] = []
    labels: list[bool] = []
    confidences: list[float] = []
    correctness: list[bool] = []

    for pred in predicted:
        pred_nodes = set(pred.node_ids)
        matched = False
        for gold_kind, gold_nodes in gold_regions:
            kind_match = gold_kind is None or pred.kind.value == gold_kind
            if kind_match and _match_anomaly(pred_nodes, gold_nodes):
                matched = True
                break

        scores.append(pred.severity)
        labels.append(matched)
        confidences.append(pred.confidence)
        correctness.append(matched)

    # Metrics
    ap = compute_average_precision(scores, labels) if scores else 0.0
    p_at_3 = compute_precision_at_k(scores, labels, k=3) if scores else 0.0
    ece = compute_ece(confidences, correctness) if confidences else 0.0
    brier = compute_brier(confidences, correctness) if confidences else 0.0
    calibration_score = 1.0 - min(1.0, ece)

    return {
        "case_id": case_id,
        "average_precision": ap,
        "precision_at_3": p_at_3,
        "ece": ece,
        "brier": brier,
        "calibration_score": calibration_score,
        "score": geometric_mean(ap, p_at_3, calibration_score),
        "n_predicted": len(predicted),
        "n_gold": len(gold_regions),
    }


def aggregate_anomaly_scores(case_results: list[dict]) -> dict:
    """Aggregate anomaly scores across cases."""
    if not case_results:
        return {"score": 0.0, "average_precision": 0.0, "precision_at_3": 0.0, "calibration_score": 0.0}

    ap = sum(r["average_precision"] for r in case_results) / len(case_results)
    pk = sum(r["precision_at_3"] for r in case_results) / len(case_results)
    cs = sum(r["calibration_score"] for r in case_results) / len(case_results)

    return {
        "score": geometric_mean(ap, pk, cs),
        "average_precision": ap,
        "precision_at_3": pk,
        "calibration_score": cs,
        "per_case": case_results,
    }
