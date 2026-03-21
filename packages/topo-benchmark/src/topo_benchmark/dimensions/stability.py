"""Stability dimension scorer."""

from __future__ import annotations

from collections import Counter
from typing import Callable

from topo_analyzer.analysis import StructuralAnalysis
from topo_parser_python.graph import CodeGraph

from topo_benchmark.scoring import compute_ari, geometric_mean
from topo_benchmark.signals import partition_labels


def _role_macro_f1(
    base_roles: dict[str, str],
    pert_roles: dict[str, str],
    node_mapping: dict[str, str],
) -> float:
    """Macro F1 of structural roles on mapped nodes."""
    mapped_base = {}
    mapped_pert = {}
    for base_id, pert_id in node_mapping.items():
        if base_id in base_roles and pert_id in pert_roles:
            mapped_base[base_id] = base_roles[base_id]
            mapped_pert[base_id] = pert_roles[pert_id]

    if not mapped_base:
        return 1.0

    all_roles = set(mapped_base.values()) | set(mapped_pert.values())
    f1_scores = []
    for role in all_roles:
        tp = sum(1 for k in mapped_base if mapped_base[k] == role and mapped_pert[k] == role)
        fp = sum(1 for k in mapped_base if mapped_base[k] != role and mapped_pert[k] == role)
        fn = sum(1 for k in mapped_base if mapped_base[k] == role and mapped_pert[k] != role)
        if tp == 0:
            f1_scores.append(0.0)
        else:
            prec = tp / (tp + fp)
            rec = tp / (tp + fn)
            f1_scores.append(2 * prec * rec / (prec + rec))

    return sum(f1_scores) / len(f1_scores) if f1_scores else 1.0


def _topk_overlap(
    base_anomalies: list,
    pert_anomalies: list,
    k: int = 5,
) -> float:
    """Overlap of top-k anomalies (by severity) between base and perturbation."""
    base_top = sorted(base_anomalies, key=lambda a: -a.severity)[:k]
    pert_top = sorted(pert_anomalies, key=lambda a: -a.severity)[:k]

    base_sets = [frozenset(a.node_ids) for a in base_top]
    pert_sets = [frozenset(a.node_ids) for a in pert_top]

    if not base_sets and not pert_sets:
        return 1.0
    if not base_sets or not pert_sets:
        return 0.0

    matches = 0
    for bs in base_sets:
        for ps in pert_sets:
            if bs & ps:  # any overlap
                matches += 1
                break

    return matches / len(base_sets)


def score_stability_case(
    case_id: str,
    base_analysis: StructuralAnalysis,
    pert_analyses: dict[str, StructuralAnalysis],
    node_mappings: dict[str, dict[str, str]],
) -> dict:
    """Score a single stability case across perturbations."""
    base_partition = partition_labels(base_analysis)
    base_roles = {ra.node_id: ra.role.value for ra in base_analysis.roles}

    partition_stabilities = []
    role_stabilities = []
    topk_stabilities = []

    for pert_name, pert_analysis in pert_analyses.items():
        mapping = node_mappings.get(pert_name, {})
        if not mapping:
            continue

        # Partition stability: ARI on mapped nodes
        pert_partition = partition_labels(pert_analysis)
        mapped_base_part = {base_id: base_partition[base_id] for base_id in mapping if base_id in base_partition}
        mapped_pert_part = {base_id: pert_partition[mapping[base_id]] for base_id in mapping if mapping[base_id] in pert_partition}
        if mapped_base_part and mapped_pert_part:
            partition_stabilities.append(max(0.0, compute_ari(mapped_base_part, mapped_pert_part)))

        # Role stability: macro F1
        pert_roles = {ra.node_id: ra.role.value for ra in pert_analysis.roles}
        role_stabilities.append(_role_macro_f1(base_roles, pert_roles, mapping))

        # Top-K stability
        topk_stabilities.append(_topk_overlap(base_analysis.anomalies, pert_analysis.anomalies))

    avg_part = sum(partition_stabilities) / len(partition_stabilities) if partition_stabilities else 1.0
    avg_role = sum(role_stabilities) / len(role_stabilities) if role_stabilities else 1.0
    avg_topk = sum(topk_stabilities) / len(topk_stabilities) if topk_stabilities else 1.0

    return {
        "case_id": case_id,
        "partition_stability": avg_part,
        "role_stability": avg_role,
        "topk_stability": avg_topk,
        "score": geometric_mean(avg_part, avg_role, avg_topk),
    }


def aggregate_stability_scores(case_results: list[dict]) -> dict:
    """Aggregate stability scores across cases."""
    if not case_results:
        return {"score": 0.0, "partition_stability": 0.0, "role_stability": 0.0, "topk_stability": 0.0}

    ps = sum(r["partition_stability"] for r in case_results) / len(case_results)
    rs = sum(r["role_stability"] for r in case_results) / len(case_results)
    ts = sum(r["topk_stability"] for r in case_results) / len(case_results)

    return {
        "score": geometric_mean(ps, rs, ts),
        "partition_stability": ps,
        "role_stability": rs,
        "topk_stability": ts,
        "per_case": case_results,
    }
