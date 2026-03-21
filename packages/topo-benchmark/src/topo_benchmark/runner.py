"""Benchmark runner — orchestrates a full benchmark run across dimensions."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from topo_analyzer.analysis import StructuralAnalysis, analyze
from topo_analyzer.projection import AnalysisLevel, AnalysisProjectionConfig
from topo_parser_python.graph import CodeGraph, EdgeKind

from topo_benchmark.baselines import directory_partition, louvain_partition
from topo_benchmark.datasets import (
    discover_cases,
    load_anomaly_case,
    load_architecture_case,
    load_metadata,
    load_mutation_case,
    load_stability_case,
)
from topo_benchmark.dimensions.anomalies import aggregate_anomaly_scores, score_anomaly_case
from topo_benchmark.dimensions.architecture import score_architecture_case
from topo_benchmark.dimensions.mutations import aggregate_mutation_scores, score_mutation_case
from topo_benchmark.dimensions.stability import aggregate_stability_scores, score_stability_case
from topo_benchmark.report import generate_summary
from topo_benchmark.scorecard import Scorecard, build_scorecard


def _make_analyze_fn(
    level: AnalysisLevel = AnalysisLevel.MODULE,
    combined: bool = True,
) -> Callable[[CodeGraph], StructuralAnalysis]:
    """Create an analysis function with the given defaults."""
    def analyze_fn(graph: CodeGraph) -> StructuralAnalysis:
        config = AnalysisProjectionConfig.for_analysis(
            edge_kind=EdgeKind.CALLS,
            combined=combined,
            level=level,
            internal_only=False,
        )
        return analyze(graph, combined=combined, projection_config=config)
    return analyze_fn


def run_benchmark(
    tier: str = "analyzer",
    split: str = "public",
    dataset_root: Path | None = None,
    output_dir: Path | None = None,
) -> Scorecard:
    """Run the full benchmark and return a scorecard.

    Args:
        tier: "analyzer" or "e2e"
        split: "public", "hidden", or "smoke"
        dataset_root: Path to benchmark/datasets/
        output_dir: Where to write artifacts. Auto-generated if None.
    """
    if output_dir is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
        output_dir = Path(".benchmark") / "runs" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    per_case_lines: list[str] = []
    failures: list[dict] = []
    dimension_results: dict[str, dict] = {}
    baseline_results: dict[str, dict] = {}

    analyze_fn = _make_analyze_fn(AnalysisLevel.MODULE)
    analyze_fn_symbol = _make_analyze_fn(AnalysisLevel.SYMBOL)

    # --- Architecture Recovery ---
    arch_cases = discover_cases("architecture", split, dataset_root)
    if arch_cases:
        arch_case_results = []
        for case_dir in arch_cases:
            try:
                graph, labels = load_architecture_case(case_dir)
                analysis = analyze_fn(graph)

                # Baselines
                dir_part = directory_partition(graph)
                try:
                    louv_part = louvain_partition(graph)
                except Exception:
                    louv_part = dir_part

                bp = {"directory": dir_part, "louvain": louv_part}
                result = score_architecture_case(graph, labels, analysis, bp)
                result["case_id"] = case_dir.name
                arch_case_results.append(result)
                per_case_lines.append(json.dumps({
                    "dimension": "architecture_recovery",
                    "case_id": case_dir.name,
                    "score": result["score"],
                }))
                baseline_results["architecture"] = {
                    "directory": result.get("baseline_scores", {}).get("directory", 0),
                    "louvain": result.get("baseline_scores", {}).get("louvain", 0),
                }
            except Exception as e:
                failures.append({"dimension": "architecture_recovery", "case": case_dir.name, "error": str(e)})

        if arch_case_results:
            # For single case, use it directly
            dimension_results["architecture_recovery"] = arch_case_results[0]
        else:
            dimension_results["architecture_recovery"] = {"score": 0.0, "guardrails": {"coverage_ok": False, "baseline_ok": False}}

    # --- Mutation Ranking ---
    mut_cases = discover_cases("mutations", split, dataset_root)
    if mut_cases:
        mut_case_results = []
        for case_dir in mut_cases:
            try:
                meta = load_metadata(case_dir)
                level = AnalysisLevel(meta.get("level", "module"))
                fn = analyze_fn if level == AnalysisLevel.MODULE else analyze_fn_symbol

                variants, expectations = load_mutation_case(case_dir)
                result = score_mutation_case(case_dir.name, variants, expectations, fn)
                mut_case_results.append(result)
                per_case_lines.append(json.dumps({
                    "dimension": "mutation_ranking",
                    "case_id": case_dir.name,
                    "score": result["score"],
                }))
            except Exception as e:
                failures.append({"dimension": "mutation_ranking", "case": case_dir.name, "error": str(e)})

        dimension_results["mutation_ranking"] = aggregate_mutation_scores(mut_case_results)

    # --- Stability ---
    stab_cases = discover_cases("stability", split, dataset_root)
    if stab_cases:
        stab_case_results = []
        for case_dir in stab_cases:
            try:
                base_graph, perturbations, mappings = load_stability_case(case_dir)
                base_analysis = analyze_fn(base_graph)
                pert_analyses = {name: analyze_fn(g) for name, g in perturbations.items()}
                result = score_stability_case(case_dir.name, base_analysis, pert_analyses, mappings)
                stab_case_results.append(result)
                per_case_lines.append(json.dumps({
                    "dimension": "stability",
                    "case_id": case_dir.name,
                    "score": result["score"],
                }))
            except Exception as e:
                failures.append({"dimension": "stability", "case": case_dir.name, "error": str(e)})

        dimension_results["stability"] = aggregate_stability_scores(stab_case_results)

    # --- Anomaly Precision & Calibration ---
    anom_cases = discover_cases("anomalies", split, dataset_root)
    if anom_cases:
        anom_case_results = []
        for case_dir in anom_cases:
            try:
                graph, gold = load_anomaly_case(case_dir)
                analysis = analyze_fn(graph)
                result = score_anomaly_case(case_dir.name, analysis, gold)
                anom_case_results.append(result)
                per_case_lines.append(json.dumps({
                    "dimension": "anomaly_precision_calibration",
                    "case_id": case_dir.name,
                    "score": result["score"],
                }))
            except Exception as e:
                failures.append({"dimension": "anomaly_precision_calibration", "case": case_dir.name, "error": str(e)})

        dimension_results["anomaly_precision_calibration"] = aggregate_anomaly_scores(anom_case_results)

    # --- Build Scorecard ---
    scorecard = build_scorecard(tier, split, dimension_results, baseline_results)

    # --- Write Artifacts ---
    scorecard.save(output_dir / "scorecard.json")

    (output_dir / "dimensions.json").write_text(
        json.dumps({k: _safe_serialize(v) for k, v in dimension_results.items()}, indent=2) + "\n"
    )

    (output_dir / "per_case.jsonl").write_text("\n".join(per_case_lines) + "\n" if per_case_lines else "")

    (output_dir / "failures.json").write_text(json.dumps(failures, indent=2) + "\n")

    baselines_dir = output_dir / "baselines"
    baselines_dir.mkdir(exist_ok=True)
    for name, results in baseline_results.items():
        (baselines_dir / f"{name}.json").write_text(json.dumps(results, indent=2) + "\n")

    summary = generate_summary(scorecard.to_dict(), dimension_results, baseline_results)
    (output_dir / "summary.md").write_text(summary)

    return scorecard


def _safe_serialize(obj: object) -> object:
    """Make dimension results JSON-serializable."""
    if isinstance(obj, dict):
        return {k: _safe_serialize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_safe_serialize(item) for item in obj]
    if isinstance(obj, float):
        return round(obj, 6)
    return obj
