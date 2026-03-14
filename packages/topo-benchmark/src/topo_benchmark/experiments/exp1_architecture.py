"""Experiment 1: Cross-Directory Architecture Recovery.

The decisive test for the core bet. Measures whether spectral analysis
recovers architectural modules that span directory boundaries — something
directory grouping cannot do by definition.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

from topo_analyzer.analysis import StructuralAnalysis, analyze
from topo_analyzer.projection import AnalysisLevel, AnalysisProjectionConfig
from topo_parser.graph import CodeGraph, EdgeKind
from topo_parser.python import parse_python_project

from topo_benchmark.baselines import directory_partition, louvain_partition
from topo_benchmark.experiments.config import CODEBASES, THRESHOLDS
from topo_benchmark.experiments.gold_labels import load_gold_labels, validate_gold_labels
from topo_benchmark.scoring import (
    compute_boundary_f1,
    compute_cross_directory_recovery,
    compute_v_measure,
)
from topo_benchmark.signals import partition_labels


@dataclass
class CodebaseResult:
    """Results for a single codebase."""

    name: str
    # V-measure per method
    spectral_v_measure: float = 0.0
    spectral_homogeneity: float = 0.0
    spectral_completeness: float = 0.0
    directory_v_measure: float = 0.0
    louvain_v_measure: float = 0.0
    # Boundary F1 per method
    spectral_boundary_f1: float = 0.0
    directory_boundary_f1: float = 0.0
    louvain_boundary_f1: float = 0.0
    # Cross-directory recovery (the decisive metric)
    spectral_cross_dir_recovery: float = 0.0
    louvain_cross_dir_recovery: float = 0.0
    # Coverage and graph stats
    spectral_coverage: float = 0.0
    graph_nodes: int = 0
    graph_edges: int = 0
    gold_label_coverage: float = 0.0
    # Validation
    label_validation: dict = field(default_factory=dict)
    # Cluster details for inspection
    spectral_clusters: dict = field(default_factory=dict)
    error: str | None = None

    @property
    def spectral_beats_directory(self) -> bool:
        return self.spectral_v_measure > self.directory_v_measure

    @property
    def spectral_beats_louvain(self) -> bool:
        return self.spectral_v_measure > self.louvain_v_measure


@dataclass
class Experiment1Result:
    """Aggregate results for Experiment 1."""

    codebase_results: list[CodebaseResult]
    # Aggregate verdicts
    cross_dir_recovery_rates: dict[str, float] = field(default_factory=dict)
    v_measure_wins_vs_directory: int = 0
    v_measure_wins_vs_louvain: int = 0
    codebases_above_cross_dir_pass: int = 0
    codebases_below_cross_dir_fail: int = 0
    # Final verdict
    verdict: str = "INCONCLUSIVE"
    verdict_details: list[str] = field(default_factory=list)


def _clone_codebase(name: str, info: dict, codebases_root: Path) -> Path:
    """Clone a codebase if not already present. Returns path to source root."""
    repo_dir = codebases_root / name
    if not repo_dir.exists():
        repo_url = f"https://github.com/{info['repo']}.git"
        ref = info.get("ref", "main")
        subprocess.run(
            ["git", "clone", "--depth", "1", "--branch", ref, repo_url, str(repo_dir)],
            check=True,
            capture_output=True,
        )
    src_root = info.get("src_root", "")
    return repo_dir / src_root if src_root else repo_dir


def _directory_labels(node_ids: set[str]) -> dict[str, str]:
    """Compute directory-level labels from node IDs.

    Uses the second-level component for hierarchical packages,
    or the top-level for flat packages.
    """
    labels = {}
    for nid in node_ids:
        parts = nid.split(".")
        if len(parts) >= 2:
            labels[nid] = ".".join(parts[:2])
        else:
            labels[nid] = parts[0]
    return labels


def _analyze_codebase(graph: CodeGraph) -> StructuralAnalysis:
    """Run spectral analysis at module level with combined layers."""
    config = AnalysisProjectionConfig.for_analysis(
        edge_kind=EdgeKind.CALLS,
        combined=True,
        level=AnalysisLevel.MODULE,
        internal_only=False,
    )
    return analyze(graph, combined=True, projection_config=config)


def _extract_cluster_details(analysis: StructuralAnalysis) -> dict:
    """Extract cluster membership for inspection."""
    clusters: dict[int, list[str]] = {}
    for module in analysis.modules:
        if not module.unassigned:
            clusters[module.id] = sorted(module.node_ids)
    return clusters


def run_single_codebase(
    name: str,
    graph: CodeGraph,
    labels_root: Path,
) -> CodebaseResult:
    """Run Experiment 1 on a single codebase."""
    result = CodebaseResult(name=name)

    # Load gold labels
    try:
        labels = load_gold_labels(name, labels_root)
    except FileNotFoundError as e:
        result.error = str(e)
        return result

    gold = labels["included_nodes"]
    gold_nodes = set(gold.keys())

    # Run spectral analysis (projects to module level internally)
    try:
        analysis = _analyze_codebase(graph)
    except Exception as e:
        result.error = f"Spectral analysis failed: {e}"
        return result

    # Validate gold labels against projected (module-level) graph
    projected_graph = analysis.graph
    validation = validate_gold_labels(labels, projected_graph)
    result.label_validation = validation
    result.gold_label_coverage = validation["coverage"]

    spectral_partition = partition_labels(analysis)
    result.spectral_clusters = _extract_cluster_details(analysis)

    # Coverage: fraction of gold nodes that got a spectral assignment
    assigned_gold = sum(1 for n in gold_nodes if n in spectral_partition)
    result.spectral_coverage = assigned_gold / len(gold_nodes) if gold_nodes else 0.0

    # Graph stats at projected level
    result.graph_nodes = len(projected_graph.nodes)
    result.graph_edges = len(projected_graph.edges)

    # Baselines on projected graph (module-level, matching gold labels)
    dir_partition = directory_partition(projected_graph)
    try:
        louv_partition = louvain_partition(projected_graph)
    except Exception:
        louv_partition = dir_partition

    # Directory labels for cross-directory metric
    dir_labels = _directory_labels(gold_nodes)

    # --- V-measure ---
    _, _, result.spectral_v_measure = compute_v_measure(gold, spectral_partition)
    result.spectral_homogeneity, result.spectral_completeness, _ = compute_v_measure(
        gold, spectral_partition
    )
    _, _, result.directory_v_measure = compute_v_measure(gold, dir_partition)
    _, _, result.louvain_v_measure = compute_v_measure(gold, louv_partition)

    # --- Boundary F1 ---
    edges = [
        (e.source, e.target) for e in projected_graph.edges
        if e.source in gold_nodes and e.target in gold_nodes
    ]

    result.spectral_boundary_f1 = compute_boundary_f1(edges, spectral_partition, gold)
    result.directory_boundary_f1 = compute_boundary_f1(edges, dir_partition, gold)
    result.louvain_boundary_f1 = compute_boundary_f1(edges, louv_partition, gold)

    # --- Cross-directory recovery (the decisive metric) ---
    result.spectral_cross_dir_recovery = compute_cross_directory_recovery(
        spectral_partition, gold, dir_labels
    )
    result.louvain_cross_dir_recovery = compute_cross_directory_recovery(
        louv_partition, gold, dir_labels
    )

    return result


def run_experiment_1(
    codebases_root: Path,
    labels_root: Path,
    output_dir: Path,
    codebase_filter: list[str] | None = None,
) -> Experiment1Result:
    """Run Experiment 1: Cross-Directory Architecture Recovery.

    Args:
        codebases_root: Directory to clone/find codebases.
        labels_root: Directory containing gold_labels/{codebase}/labels.json.
        output_dir: Where to write results.
        codebase_filter: If provided, only run these codebases.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    thresholds = THRESHOLDS["exp1"]

    codebases = CODEBASES
    if codebase_filter:
        codebases = {k: v for k, v in codebases.items() if k in codebase_filter}

    codebase_results: list[CodebaseResult] = []

    for name, info in codebases.items():
        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")

        # Clone/find codebase
        try:
            src_path = _clone_codebase(name, info, codebases_root)
            print(f"  Source: {src_path}")
        except Exception as e:
            cr = CodebaseResult(name=name, error=f"Clone failed: {e}")
            codebase_results.append(cr)
            continue

        # Parse
        try:
            graph = parse_python_project(
                src_path.parent if info.get("src_root") else src_path,
                exclude_patterns=["tests", ".venv", "test", "examples", "docs"],
            )
            print(f"  Parsed: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
        except Exception as e:
            cr = CodebaseResult(name=name, error=f"Parse failed: {e}")
            codebase_results.append(cr)
            continue

        # Run experiment
        cr = run_single_codebase(name, graph, labels_root)
        codebase_results.append(cr)

        if cr.error:
            print(f"  ERROR: {cr.error}")
            continue

        # Print summary
        print(f"  Gold label coverage: {cr.gold_label_coverage:.1%}")
        print(f"  Spectral coverage:   {cr.spectral_coverage:.1%}")
        print(f"  V-measure:  spectral={cr.spectral_v_measure:.3f}  "
              f"directory={cr.directory_v_measure:.3f}  "
              f"louvain={cr.louvain_v_measure:.3f}")
        print(f"  Boundary F1: spectral={cr.spectral_boundary_f1:.3f}  "
              f"directory={cr.directory_boundary_f1:.3f}  "
              f"louvain={cr.louvain_boundary_f1:.3f}")

        import math
        if not math.isnan(cr.spectral_cross_dir_recovery):
            print(f"  Cross-dir recovery: spectral={cr.spectral_cross_dir_recovery:.1%}  "
                  f"louvain={cr.louvain_cross_dir_recovery:.1%}")
        else:
            print(f"  Cross-dir recovery: N/A (no cross-directory pairs in gold labels)")

        print(f"  Spectral beats directory: {cr.spectral_beats_directory}")
        print(f"  Spectral beats Louvain:   {cr.spectral_beats_louvain}")

    # --- Aggregate ---
    experiment_result = Experiment1Result(codebase_results=codebase_results)

    valid_results = [cr for cr in codebase_results if cr.error is None]

    for cr in valid_results:
        experiment_result.cross_dir_recovery_rates[cr.name] = cr.spectral_cross_dir_recovery
        if cr.spectral_beats_directory:
            experiment_result.v_measure_wins_vs_directory += 1
        if cr.spectral_beats_louvain:
            experiment_result.v_measure_wins_vs_louvain += 1

    import math
    cross_dir_results = [
        cr for cr in valid_results
        if not math.isnan(cr.spectral_cross_dir_recovery)
    ]
    experiment_result.codebases_above_cross_dir_pass = sum(
        1 for cr in cross_dir_results
        if cr.spectral_cross_dir_recovery >= thresholds["cross_dir_recovery_pass"]
    )
    experiment_result.codebases_below_cross_dir_fail = sum(
        1 for cr in cross_dir_results
        if cr.spectral_cross_dir_recovery < thresholds["cross_dir_recovery_fail"]
    )

    # --- Verdict ---
    details = []
    n_valid = len(valid_results)
    n_cross_dir = len(cross_dir_results)

    if n_valid == 0:
        experiment_result.verdict = "ERROR"
        details.append("No codebases produced results.")
    else:
        # Check cross-directory recovery
        if n_cross_dir > 0:
            pass_rate = experiment_result.codebases_above_cross_dir_pass
            fail_rate = experiment_result.codebases_below_cross_dir_fail
            details.append(
                f"Cross-dir recovery: {pass_rate}/{n_cross_dir} above {thresholds['cross_dir_recovery_pass']:.0%} threshold, "
                f"{fail_rate}/{n_cross_dir} below {thresholds['cross_dir_recovery_fail']:.0%} threshold"
            )

        # Check V-measure wins
        details.append(
            f"V-measure vs directory: spectral wins {experiment_result.v_measure_wins_vs_directory}/{n_valid}"
        )
        details.append(
            f"V-measure vs Louvain: spectral wins {experiment_result.v_measure_wins_vs_louvain}/{n_valid}"
        )

        # Determine verdict
        # With only 2 codebases, we can't meet the 3/5 threshold yet.
        # Report what we have and flag as preliminary.
        min_pass = thresholds["min_codebases_pass"]
        min_total = thresholds["min_codebases_total"]

        if n_valid < min_total:
            experiment_result.verdict = "PRELIMINARY"
            details.append(
                f"Only {n_valid}/{min_total} codebases tested. "
                f"Need {min_total} for definitive verdict."
            )
        elif (
            experiment_result.codebases_above_cross_dir_pass >= min_pass
            and experiment_result.v_measure_wins_vs_directory >= min_pass
            and experiment_result.v_measure_wins_vs_louvain >= min_pass
        ):
            experiment_result.verdict = "PASS"
            details.append("All thresholds met.")
        elif experiment_result.codebases_below_cross_dir_fail >= min_pass:
            experiment_result.verdict = "FAIL"
            details.append(
                f"Cross-directory recovery below {thresholds['cross_dir_recovery_fail']:.0%} "
                f"on {experiment_result.codebases_below_cross_dir_fail}/{n_valid} codebases."
            )
        elif experiment_result.v_measure_wins_vs_directory < min_pass:
            experiment_result.verdict = "FAIL"
            details.append(
                f"Spectral V-measure beats directory on only "
                f"{experiment_result.v_measure_wins_vs_directory}/{n_valid} codebases "
                f"(need {min_pass})."
            )
        else:
            experiment_result.verdict = "INCONCLUSIVE"
            details.append("Mixed results — some thresholds met, others not.")

    experiment_result.verdict_details = details

    # --- Write results ---
    results_dict = {
        "experiment": "exp1_cross_directory_architecture_recovery",
        "verdict": experiment_result.verdict,
        "verdict_details": experiment_result.verdict_details,
        "thresholds": thresholds,
        "codebases": [asdict(cr) for cr in codebase_results],
        "aggregate": {
            "cross_dir_recovery_rates": experiment_result.cross_dir_recovery_rates,
            "v_measure_wins_vs_directory": experiment_result.v_measure_wins_vs_directory,
            "v_measure_wins_vs_louvain": experiment_result.v_measure_wins_vs_louvain,
            "codebases_above_cross_dir_pass": experiment_result.codebases_above_cross_dir_pass,
            "codebases_below_cross_dir_fail": experiment_result.codebases_below_cross_dir_fail,
        },
    }
    (output_dir / "exp1_results.json").write_text(
        json.dumps(results_dict, indent=2, default=str) + "\n"
    )

    return experiment_result
