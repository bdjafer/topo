"""Mutation ranking dimension scorer."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from topo_analyzer.analysis import StructuralAnalysis
from topo_parser_python.graph import CodeGraph

from topo_benchmark import signals
from topo_benchmark.scoring import geometric_mean


def _eval_signal(
    analyses: dict[str, StructuralAnalysis],
    expectation: dict,
) -> bool:
    """Evaluate a single expectation against analysis results."""
    signal_name = expectation["signal"]
    direction = expectation.get("direction", "")
    margin = expectation.get("margin", 0.0)

    variants = expectation.get("variants")
    if variants:
        first_name, second_name = variants
        first = analyses[first_name]
        second = analyses[second_name]
    else:
        return False

    args = expectation.get("signal_args", {})

    if signal_name == "max_cross_module_severity" or signal_name == "max_cycle_severity":
        kind = args.get("kind", signal_name.replace("max_", "").replace("_severity", ""))
        val_first = signals.max_anomaly_severity(first, kind)
        val_second = signals.max_anomaly_severity(second, kind)
        if direction == "higher_in_second":
            return val_second > val_first + margin
        return False

    if signal_name == "attribution_at_3":
        mutated_region = set(expectation.get("mutated_region", {}).get("nodes", []))
        if not mutated_region:
            return False
        if direction == "true_in_second":
            return signals.attribution_at_k(second, mutated_region, k=3)
        return False

    if signal_name == "partition_similarity_to_clean":
        clean = analyses.get("clean")
        if clean is None:
            return False
        sim = signals.partition_similarity(second, clean)
        if direction == "lower_in_second":
            sim_first = signals.partition_similarity(first, clean) if first is not clean else 1.0
            return sim < sim_first - margin
        return False

    if signal_name == "largest_module_ratio":
        val_first = signals.largest_module_ratio(first)
        val_second = signals.largest_module_ratio(second)
        if direction == "higher_in_second":
            return val_second > val_first + margin
        return False

    if signal_name == "target_role":
        node_id = args.get("node_id", "")
        expected_roles = expectation.get("expected_roles", [])
        role = signals.target_role(second, node_id)
        return role in expected_roles if role else False

    if signal_name == "target_role_change":
        node_id = args.get("node_id", "")
        clean_role_expected = expectation.get("clean_role")
        role_clean = signals.target_role(first, node_id)
        role_mutated = signals.target_role(second, node_id)
        if direction == "role_changes":
            return role_clean == clean_role_expected and role_mutated != clean_role_expected
        return False

    if signal_name == "has_finding":
        kind = args.get("kind", "")
        if direction == "true_in_second":
            return signals.has_finding(second, kind)
        if direction == "false_in_first":
            return not signals.has_finding(first, kind)
        if direction == "present_in_second_not_first":
            return signals.has_finding(second, kind) and not signals.has_finding(first, kind)
        return False

    if signal_name == "cross_package_dep_count":
        val_first = signals.cross_package_dep_count(first)
        val_second = signals.cross_package_dep_count(second)
        if direction == "higher_in_second":
            return val_second > val_first
        return False

    if signal_name == "has_cross_package_dep":
        src = args.get("source_pkg", "")
        tgt = args.get("target_pkg", "")
        if direction == "true_in_second":
            return signals.has_cross_package_dep(second, src, tgt)
        if direction == "absent_in_first":
            return not signals.has_cross_package_dep(first, src, tgt)
        return False

    return False


def score_mutation_case(
    case_id: str,
    variants: dict[str, CodeGraph],
    expectations: dict,
    analyze_fn: Callable[[CodeGraph], StructuralAnalysis],
) -> dict:
    """Score a single mutation ranking case."""
    # Run analysis on all variants
    analyses: dict[str, StructuralAnalysis] = {}
    for name, graph in variants.items():
        analyses[name] = analyze_fn(graph)

    # Get the mutated_region from expectations
    mutated_region = expectations.get("mutated_region", {})

    # Evaluate pairwise ordering
    ordering = expectations.get("ordering", [])
    pairwise_results: list[bool] = []
    for better, worse in ordering:
        if better in analyses and worse in analyses:
            # Find expectations matching this pair
            matched_exps = [
                exp for exp in expectations.get("required_expectations", [])
                if exp.get("variants") and list(exp["variants"]) == [better, worse]
            ]
            # Pairs with no matching expectations FAIL — no free passes
            if not matched_exps:
                pairwise_results.append(False)
                continue
            # All matched expectations must pass
            pair_pass = all(
                _eval_signal(analyses, {**exp, "mutated_region": mutated_region})
                for exp in matched_exps
            )
            pairwise_results.append(pair_pass)

    pairwise_accuracy = sum(pairwise_results) / len(pairwise_results) if pairwise_results else 0.0

    # Repair accuracy: repaired should be closer to clean than mutated.
    # Only evaluated for cases that have a repaired variant.
    # Cases without repair are excluded (not counted), not given a free 1.0.
    has_repair = "repaired" in analyses
    repair_results: list[bool] = []
    if has_repair and "clean" in analyses and "mutated" in analyses:
        sim_repaired = signals.partition_similarity(analyses["repaired"], analyses["clean"])
        sim_mutated = signals.partition_similarity(analyses["mutated"], analyses["clean"])
        repair_results.append(sim_repaired >= sim_mutated)

    repair_accuracy: float | None = None
    if repair_results:
        repair_accuracy = sum(repair_results) / len(repair_results)

    # Attribution: does the mutated region appear in top anomalies?
    attr_results: list[bool] = []
    region_nodes = set(mutated_region.get("nodes", []))
    if region_nodes and "mutated" in analyses:
        attr_results.append(signals.attribution_at_k(analyses["mutated"], region_nodes, k=3))
    attribution = sum(attr_results) / len(attr_results) if attr_results else 0.0

    # Score: geometric mean of available components.
    # Repair accuracy is excluded (not set to 1.0) when no repair variant exists.
    # Attribution of 0 means 0 — no silent inflation.
    score_components = [pairwise_accuracy, attribution]
    if repair_accuracy is not None:
        score_components.append(repair_accuracy)
    score = geometric_mean(*score_components)

    return {
        "case_id": case_id,
        "pairwise_accuracy": pairwise_accuracy,
        "repair_accuracy": repair_accuracy,
        "attribution_at_3": attribution,
        "has_repair_variant": has_repair,
        "score": score,
        "pairwise_details": pairwise_results,
    }


def aggregate_mutation_scores(case_results: list[dict]) -> dict:
    """Aggregate mutation scores across all cases."""
    if not case_results:
        return {"score": 0.0, "pairwise_accuracy": 0.0, "repair_accuracy": 0.0, "attribution_at_3": 0.0}

    pw = sum(r["pairwise_accuracy"] for r in case_results) / len(case_results)
    at = sum(r["attribution_at_3"] for r in case_results) / len(case_results)

    # Repair accuracy: only average over cases that have a repair variant.
    # Cases without repair are excluded, not counted as 1.0.
    repair_cases = [r for r in case_results if r["repair_accuracy"] is not None]
    ra: float | None = None
    if repair_cases:
        ra = sum(r["repair_accuracy"] for r in repair_cases) / len(repair_cases)

    score_components = [pw, at]
    if ra is not None:
        score_components.append(ra)

    return {
        "score": geometric_mean(*score_components),
        "pairwise_accuracy": pw,
        "repair_accuracy": ra,
        "attribution_at_3": at,
        "per_case": case_results,
    }
