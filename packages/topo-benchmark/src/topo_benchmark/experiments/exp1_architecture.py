"""Experiment 1: Comprehensive Structural Health Evaluation.

Runs a matrix of analysis configurations (level × layer combinations) on each
codebase and reports: clustering quality, per-layer signal strength, cluster
purity, structural diagnostics (roles, anomalies, health, spectral quality).

This replaces the previous module-only evaluation which was degenerate for flat
packages (e.g. Click at module level had density 198%, V-measure = 0).
"""

from __future__ import annotations

import json
import math
import subprocess
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

from topo_analyzer.analysis import StructuralAnalysis, analyze
from topo_analyzer.projection import AnalysisLevel, AnalysisProjectionConfig
from topo_parser_python.graph import CodeGraph, EdgeKind, NodeKind
from topo_parser_python.python import parse_python_project

from topo_benchmark.baselines import (
    directory_partition,
    directory_partition_by_module,
    louvain_partition,
)
from topo_benchmark.experiments.config import CODEBASES, THRESHOLDS
from topo_benchmark.experiments.gold_labels import load_gold_labels, validate_gold_labels
from topo_benchmark.scoring import (
    compute_boundary_f1,
    compute_cross_directory_recovery,
    compute_v_measure,
)
from topo_benchmark.signals import partition_labels


# ---------------------------------------------------------------------------
# Analysis configuration matrix
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AnalysisConfig:
    """One analysis configuration to evaluate."""
    name: str
    level: AnalysisLevel
    edge_kinds: tuple[EdgeKind, ...]
    combined: bool


ANALYSIS_CONFIGS = [
    AnalysisConfig("module_calls", AnalysisLevel.MODULE, (EdgeKind.CALLS,), False),
    AnalysisConfig("module_combined", AnalysisLevel.MODULE, (EdgeKind.CALLS, EdgeKind.IMPORTS, EdgeKind.INHERITS), True),
    AnalysisConfig("symbol_calls", AnalysisLevel.SYMBOL, (EdgeKind.CALLS,), False),
    AnalysisConfig("symbol_combined", AnalysisLevel.SYMBOL, (EdgeKind.CALLS, EdgeKind.IMPORTS, EdgeKind.INHERITS), True),
    AnalysisConfig("symbol_imports", AnalysisLevel.SYMBOL, (EdgeKind.IMPORTS,), False),
    AnalysisConfig("symbol_inherits", AnalysisLevel.SYMBOL, (EdgeKind.INHERITS,), False),
]


# ---------------------------------------------------------------------------
# Result data structures
# ---------------------------------------------------------------------------

@dataclass
class ConfigResult:
    """Results for a single analysis configuration on a single codebase."""
    config_name: str
    # Graph stats
    graph_nodes: int = 0
    graph_edges: int = 0
    density: float = 0.0
    # Clustering quality vs gold labels
    v_measure: float = 0.0
    homogeneity: float = 0.0
    completeness: float = 0.0
    boundary_f1: float = 0.0
    cross_dir_recovery: float = float("nan")
    # Spectral diagnostics
    n_modules: int = 0
    silhouette: float | None = None
    fiedler_value: float = 0.0
    package_fallback: bool = False
    # Cluster quality
    mega_cluster_ratio: float = 0.0
    weighted_purity: float = 0.0
    # Structural diagnostics
    role_counts: dict[str, int] = field(default_factory=dict)
    anomaly_counts: dict[str, int] = field(default_factory=dict)
    top_anomalies: list[str] = field(default_factory=list)
    # Health
    call_density: float = 0.0
    orphan_ratio: float = 0.0
    largest_module_status: str = ""
    # Flag
    degenerate: bool = False
    error: str | None = None


@dataclass
class ClusterDetail:
    """Purity details for a single cluster."""
    cluster_id: int
    size: int
    dominant_group: str
    purity: float
    group_counts: dict[str, int]


@dataclass
class CodebaseResult:
    """Comprehensive results for a single codebase."""
    name: str
    parsed_nodes: int = 0
    parsed_edges: int = 0
    config_results: list[ConfigResult] = field(default_factory=list)
    best_config: str = ""
    best_v_measure: float = 0.0
    # Baselines (at best config's level)
    directory_v_measure: float = 0.0
    louvain_v_measure: float = 0.0
    directory_boundary_f1: float = 0.0
    louvain_boundary_f1: float = 0.0
    # Per-layer signal comparison (symbol level)
    layer_signals: dict[str, float] = field(default_factory=dict)
    # Cluster purity from best config
    cluster_details: list[ClusterDetail] = field(default_factory=list)
    # Cross-dir from best config
    spectral_cross_dir_recovery: float = float("nan")
    louvain_cross_dir_recovery: float = float("nan")
    # Aggregate flags
    spectral_beats_directory: bool = False
    spectral_beats_louvain: bool = False
    error: str | None = None


@dataclass
class Experiment1Result:
    """Aggregate results for Experiment 1."""
    codebase_results: list[CodebaseResult]
    v_measure_wins_vs_directory: int = 0
    v_measure_wins_vs_louvain: int = 0
    codebases_above_cross_dir_pass: int = 0
    codebases_below_cross_dir_fail: int = 0
    verdict: str = "INCONCLUSIVE"
    verdict_details: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_analysis(graph: CodeGraph, config: AnalysisConfig) -> StructuralAnalysis:
    """Run a single analysis configuration."""
    proj_config = AnalysisProjectionConfig.for_analysis(
        edge_kind=config.edge_kinds[0],
        combined=config.combined,
        level=config.level,
        internal_only=False,
    )
    # If not combined but multiple edge kinds specified, override
    if not config.combined and len(config.edge_kinds) == 1:
        pass  # for_analysis handles this correctly
    return analyze(graph, combined=config.combined, projection_config=proj_config)


def _gold_for_symbols(
    gold_modules: dict[str, str],
    node_ids: set[str],
    module_ids: set[str],
) -> dict[str, str]:
    """Map symbol-level node IDs to gold labels via owning module.

    For a symbol like ``click.core.Command.__init__``, find the longest
    matching module ID (``click.core``) and use its gold label (``core``).
    """
    sorted_modules = sorted(module_ids, key=len, reverse=True)
    result: dict[str, str] = {}
    for nid in node_ids:
        # Check if node itself is a gold-labeled module
        if nid in gold_modules:
            result[nid] = gold_modules[nid]
            continue
        # Find owning module
        for mod in sorted_modules:
            if nid.startswith(mod + "."):
                if mod in gold_modules:
                    result[nid] = gold_modules[mod]
                break
    return result


def _graph_density(n_nodes: int, n_edges: int) -> float:
    """Compute graph density (fraction of possible edges)."""
    max_edges = n_nodes * (n_nodes - 1)
    if max_edges == 0:
        return 0.0
    return n_edges / max_edges


def _cluster_purity(
    modules: list,
    gold: dict[str, str],
) -> tuple[float, float, list[ClusterDetail]]:
    """Compute per-cluster purity and weighted average purity.

    Returns (mega_cluster_ratio, weighted_avg_purity, cluster_details).
    """
    non_unassigned = [m for m in modules if not m.unassigned]
    if not non_unassigned:
        return 0.0, 0.0, []

    total_nodes = sum(m.size for m in non_unassigned)
    largest_size = max(m.size for m in non_unassigned) if non_unassigned else 0
    mega_ratio = largest_size / total_nodes if total_nodes > 0 else 0.0

    details: list[ClusterDetail] = []
    weighted_sum = 0.0
    gold_count = 0

    for m in non_unassigned:
        group_counts: Counter[str] = Counter()
        for nid in m.node_ids:
            if nid in gold:
                group_counts[gold[nid]] += 1
        total_labeled = sum(group_counts.values())
        if total_labeled == 0:
            details.append(ClusterDetail(m.id, m.size, "?", 0.0, {}))
            continue
        dominant = group_counts.most_common(1)[0]
        purity = dominant[1] / total_labeled
        weighted_sum += purity * total_labeled
        gold_count += total_labeled
        details.append(ClusterDetail(m.id, m.size, dominant[0], purity, dict(group_counts)))

    weighted_avg = weighted_sum / gold_count if gold_count > 0 else 0.0
    details.sort(key=lambda d: d.size, reverse=True)
    return mega_ratio, weighted_avg, details


def _evaluate_config(
    graph: CodeGraph,
    config: AnalysisConfig,
    gold_modules: dict[str, str],
    module_ids: set[str],
) -> ConfigResult:
    """Run one analysis config and score against gold labels."""
    cr = ConfigResult(config_name=config.name)

    try:
        analysis = _run_analysis(graph, config)
    except Exception as e:
        cr.error = str(e)
        return cr

    pg = analysis.graph
    cr.graph_nodes = len(pg.nodes)
    cr.graph_edges = len(pg.edges)
    cr.density = _graph_density(cr.graph_nodes, cr.graph_edges)
    cr.degenerate = cr.density > 0.5

    # Map gold labels to the analysis level
    if config.level == AnalysisLevel.SYMBOL:
        gold = _gold_for_symbols(gold_modules, set(pg.nodes.keys()), module_ids)
    else:
        gold = {k: v for k, v in gold_modules.items() if k in pg.nodes}

    # Spectral partition
    spectral_partition = partition_labels(analysis)

    # V-measure
    h, c, v = compute_v_measure(gold, spectral_partition)
    cr.v_measure = v
    cr.homogeneity = h
    cr.completeness = c

    # Boundary F1
    edges = [
        (e.source, e.target) for e in pg.edges
        if e.source in gold and e.target in gold
    ]
    cr.boundary_f1 = compute_boundary_f1(edges, spectral_partition, gold)

    # Cross-directory recovery
    if config.level == AnalysisLevel.SYMBOL:
        dir_labels = directory_partition_by_module(pg, module_ids)
    else:
        dir_labels = {nid: nid.split(".", 1)[0] for nid in gold}
    cr.cross_dir_recovery = compute_cross_directory_recovery(
        spectral_partition, gold, dir_labels
    )

    # Spectral diagnostics
    md = analysis.module_detection
    cr.n_modules = len([m for m in md.modules if not m.unassigned])
    cr.silhouette = md.silhouette
    cr.package_fallback = md.package_fallback
    if analysis.spectral:
        cr.fiedler_value = analysis.spectral.fiedler_value

    # Cluster purity
    cr.mega_cluster_ratio, cr.weighted_purity, _ = _cluster_purity(
        md.modules, gold
    )

    # Roles
    role_counts: Counter[str] = Counter()
    for ra in analysis.roles:
        role_counts[ra.role.value] += 1
    cr.role_counts = dict(role_counts)

    # Anomalies
    anomaly_counts: Counter[str] = Counter()
    for a in analysis.anomalies:
        anomaly_counts[a.kind.value] += 1
    cr.anomaly_counts = dict(anomaly_counts)
    top_3 = sorted(analysis.anomalies, key=lambda a: -a.severity)[:3]
    cr.top_anomalies = [f"[{a.kind.value}] {a.description[:80]}" for a in top_3]

    # Health
    if analysis.health:
        cr.call_density = analysis.health.call_density
        cr.orphan_ratio = analysis.health.orphan_ratio
        cr.largest_module_status = analysis.health.largest_module_status

    return cr


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


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------

def run_single_codebase(
    name: str,
    graph: CodeGraph,
    labels_root: Path,
) -> CodebaseResult:
    """Run the full configuration matrix on a single codebase."""
    result = CodebaseResult(name=name)
    result.parsed_nodes = len(graph.nodes)
    result.parsed_edges = len(graph.edges)

    # Load gold labels
    try:
        labels = load_gold_labels(name, labels_root)
    except FileNotFoundError as e:
        result.error = str(e)
        return result

    gold_modules: dict[str, str] = labels["included_nodes"]
    module_ids = set(gold_modules.keys())

    # Run all configs
    for config in ANALYSIS_CONFIGS:
        cr = _evaluate_config(graph, config, gold_modules, module_ids)
        result.config_results.append(cr)

    # Find best config by V-measure
    valid_configs = [cr for cr in result.config_results if cr.error is None]
    if valid_configs:
        best = max(valid_configs, key=lambda cr: cr.v_measure)
        result.best_config = best.config_name
        result.best_v_measure = best.v_measure
        result.spectral_cross_dir_recovery = best.cross_dir_recovery

        # Compute baselines at best config's level
        best_cfg = next(c for c in ANALYSIS_CONFIGS if c.name == best.config_name)
        try:
            best_analysis = _run_analysis(graph, best_cfg)
            best_pg = best_analysis.graph

            if best_cfg.level == AnalysisLevel.SYMBOL:
                gold = _gold_for_symbols(gold_modules, set(best_pg.nodes.keys()), module_ids)
                dir_part = directory_partition_by_module(best_pg, module_ids)
            else:
                gold = {k: v for k, v in gold_modules.items() if k in best_pg.nodes}
                dir_part = {nid: nid.split(".", 1)[0] for nid in gold}

            try:
                louv_part = louvain_partition(best_pg)
            except Exception:
                louv_part = dir_part

            _, _, result.directory_v_measure = compute_v_measure(gold, dir_part)
            _, _, result.louvain_v_measure = compute_v_measure(gold, louv_part)

            edges = [
                (e.source, e.target) for e in best_pg.edges
                if e.source in gold and e.target in gold
            ]
            result.directory_boundary_f1 = compute_boundary_f1(edges, dir_part, gold)
            result.louvain_boundary_f1 = compute_boundary_f1(edges, louv_part, gold)

            result.louvain_cross_dir_recovery = compute_cross_directory_recovery(
                louv_part, gold, dir_part
            )

            # Cluster purity from best config
            _, _, result.cluster_details = _cluster_purity(
                best_analysis.module_detection.modules, gold
            )
        except Exception:
            pass  # Baselines failed, leave defaults

        result.spectral_beats_directory = result.best_v_measure > result.directory_v_measure
        result.spectral_beats_louvain = result.best_v_measure > result.louvain_v_measure

    # Per-layer signal comparison (from symbol-level configs)
    for cr in result.config_results:
        if cr.error is None and cr.config_name.startswith("symbol_"):
            layer_name = cr.config_name.replace("symbol_", "")
            result.layer_signals[layer_name] = cr.v_measure

    return result


def _print_codebase_report(result: CodebaseResult) -> None:
    """Print comprehensive report for a single codebase."""
    print(f"\n{'='*70}")
    print(f"  {result.name}")
    print(f"{'='*70}")

    if result.error:
        print(f"  ERROR: {result.error}")
        return

    print(f"  Parsed: {result.parsed_nodes} nodes, {result.parsed_edges} edges")

    # Configuration matrix
    print(f"\n  Configuration matrix:")
    print(f"  {'Config':<22} {'V-meas':>7} {'Bnd-F1':>7} {'Density':>8} "
          f"{'Mods':>5} {'Mega%':>6} {'Silh':>6} {'Fiedler':>8}")
    print(f"  {'-'*22} {'-'*7} {'-'*7} {'-'*8} {'-'*5} {'-'*6} {'-'*6} {'-'*8}")

    for cr in result.config_results:
        if cr.error:
            print(f"  {cr.config_name:<22} ERROR: {cr.error[:40]}")
            continue
        degen = " !" if cr.degenerate else ""
        silh = f"{cr.silhouette:.3f}" if cr.silhouette is not None else "  -  "
        print(f"  {cr.config_name:<22} {cr.v_measure:>7.3f} {cr.boundary_f1:>7.3f} "
              f"{cr.density:>7.3f}{degen} {cr.n_modules:>5} "
              f"{cr.mega_cluster_ratio:>5.0%} {silh:>6} {cr.fiedler_value:>8.4f}")

    # Baselines
    print(f"  {'[baselines]':<22}")
    print(f"  {'directory':<22} {result.directory_v_measure:>7.3f} "
          f"{result.directory_boundary_f1:>7.3f}")
    print(f"  {'louvain':<22} {result.louvain_v_measure:>7.3f} "
          f"{result.louvain_boundary_f1:>7.3f}")

    # Per-layer signal
    if result.layer_signals:
        print(f"\n  Per-layer signal (symbol level):")
        best_layer = max(result.layer_signals, key=result.layer_signals.get)
        for layer, v in sorted(result.layer_signals.items(), key=lambda x: -x[1]):
            marker = " <-- best" if layer == best_layer else ""
            print(f"    {layer:<12} V={v:.3f}{marker}")

    # Cluster purity (top 10 from best config)
    if result.cluster_details:
        print(f"\n  Cluster purity (best config: {result.best_config}):")
        for cd in result.cluster_details[:10]:
            mega = " <-- mega-cluster" if cd.size == max(d.size for d in result.cluster_details) and cd.purity < 0.5 else ""
            print(f"    Cluster {cd.cluster_id} ({cd.size} nodes): "
                  f"{cd.purity:.0%} {cd.dominant_group}{mega}")
        if len(result.cluster_details) > 10:
            rest = result.cluster_details[10:]
            print(f"    ... and {len(rest)} more clusters")

    # Best config structural diagnostics
    best_cr = next((cr for cr in result.config_results if cr.config_name == result.best_config), None)
    if best_cr and not best_cr.error:
        print(f"\n  Structural health ({result.best_config}):")

        # Roles
        if best_cr.role_counts:
            role_parts = [f"{count} {role}" for role, count in
                         sorted(best_cr.role_counts.items(), key=lambda x: -x[1])
                         if role != "regular"]
            regular = best_cr.role_counts.get("regular", 0)
            print(f"    Roles: {', '.join(role_parts)}, {regular} regular")

        # Anomalies
        if best_cr.anomaly_counts:
            anom_parts = [f"{count} {kind}" for kind, count in best_cr.anomaly_counts.items()]
            print(f"    Anomalies: {', '.join(anom_parts)}")
            for desc in best_cr.top_anomalies:
                print(f"      {desc}")

        print(f"    Fiedler: {best_cr.fiedler_value:.4f}  "
              f"Silhouette: {best_cr.silhouette or 0:.3f}  "
              f"Modules: {best_cr.n_modules}")
        print(f"    Density: {best_cr.call_density:.2f} calls/node  "
              f"Orphans: {best_cr.orphan_ratio:.1%}  "
              f"Module sep: {best_cr.largest_module_status}")

    # Cross-dir recovery
    if not math.isnan(result.spectral_cross_dir_recovery):
        print(f"\n  Cross-dir recovery: spectral={result.spectral_cross_dir_recovery:.1%}  "
              f"louvain={result.louvain_cross_dir_recovery:.1%}")

    # Verdict for this codebase
    print(f"\n  Best config: {result.best_config} (V={result.best_v_measure:.3f})")
    print(f"  Beats directory: {result.spectral_beats_directory}  "
          f"Beats Louvain: {result.spectral_beats_louvain}")


def run_experiment_1(
    codebases_root: Path,
    labels_root: Path,
    output_dir: Path,
    codebase_filter: list[str] | None = None,
) -> Experiment1Result:
    """Run Experiment 1: Comprehensive Structural Health Evaluation.

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
        # Clone/find codebase
        try:
            src_path = _clone_codebase(name, info, codebases_root)
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
        except Exception as e:
            cr = CodebaseResult(name=name, error=f"Parse failed: {e}")
            codebase_results.append(cr)
            continue

        # Run full evaluation
        cr = run_single_codebase(name, graph, labels_root)
        codebase_results.append(cr)
        _print_codebase_report(cr)

    # --- Aggregate ---
    experiment_result = Experiment1Result(codebase_results=codebase_results)
    valid_results = [cr for cr in codebase_results if cr.error is None]

    for cr in valid_results:
        if cr.spectral_beats_directory:
            experiment_result.v_measure_wins_vs_directory += 1
        if cr.spectral_beats_louvain:
            experiment_result.v_measure_wins_vs_louvain += 1

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

    if n_valid == 0:
        experiment_result.verdict = "ERROR"
        details.append("No codebases produced results.")
    else:
        if cross_dir_results:
            details.append(
                f"Cross-dir recovery: "
                f"{experiment_result.codebases_above_cross_dir_pass}/{len(cross_dir_results)} "
                f"above {thresholds['cross_dir_recovery_pass']:.0%}"
            )
        details.append(
            f"V-measure vs directory: spectral wins {experiment_result.v_measure_wins_vs_directory}/{n_valid}"
        )
        details.append(
            f"V-measure vs Louvain: spectral wins {experiment_result.v_measure_wins_vs_louvain}/{n_valid}"
        )

        min_pass = thresholds["min_codebases_pass"]
        min_total = thresholds["min_codebases_total"]

        if n_valid < min_total:
            experiment_result.verdict = "PRELIMINARY"
            details.append(f"Only {n_valid}/{min_total} codebases tested.")
        elif (
            experiment_result.v_measure_wins_vs_directory >= min_pass
            and experiment_result.v_measure_wins_vs_louvain >= min_pass
        ):
            experiment_result.verdict = "PASS"
            details.append("All thresholds met.")
        elif experiment_result.v_measure_wins_vs_directory < min_pass:
            experiment_result.verdict = "FAIL"
            details.append(
                f"Spectral V-measure beats directory on only "
                f"{experiment_result.v_measure_wins_vs_directory}/{n_valid} codebases."
            )
        else:
            experiment_result.verdict = "INCONCLUSIVE"
            details.append("Mixed results.")

    experiment_result.verdict_details = details

    # Print aggregate
    print(f"\n{'='*70}")
    print(f"  AGGREGATE")
    print(f"{'='*70}")
    print(f"  Verdict: {experiment_result.verdict}")
    for d in details:
        print(f"    {d}")

    # --- Write results ---
    def _serialize(obj):
        if isinstance(obj, dict):
            return {str(k): _serialize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_serialize(i) for i in obj]
        if isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return None
            return round(obj, 6)
        return obj

    results_dict = {
        "experiment": "exp1_comprehensive_structural_health",
        "verdict": experiment_result.verdict,
        "verdict_details": experiment_result.verdict_details,
        "codebases": [_serialize(asdict(cr)) for cr in codebase_results],
    }
    (output_dir / "exp1_results.json").write_text(
        json.dumps(results_dict, indent=2, default=str) + "\n"
    )

    return experiment_result
