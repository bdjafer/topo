"""
Deep investigation: does topo produce meaningful structural intelligence on real codebases?

For each codebase (Flask, Requests, Click, FastAPI), this script:
1. Parses the call graph and reports density
2. Runs full spectral analysis at module level
3. Compares detected modules against known package/file structure
4. Reports on roles, anomalies, findings
5. Provides detailed diagnostics for root cause analysis
"""

from __future__ import annotations

import sys
from collections import defaultdict
from math import comb, log
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages" / "topo-parser" / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages" / "topo-analyzer" / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages" / "pycg" / "src"))

from topo_parser.graph import CodeGraph, EdgeKind, NodeKind
from topo_parser.python import parse_python_project
from topo_analyzer.analysis import analyze
from topo_analyzer.projection import AnalysisLevel, AnalysisProjectionConfig


def compute_nmi(labels_a: list[object], labels_b: list[object]) -> float:
    n = len(labels_a)
    assert n == len(labels_b) and n > 0
    joint: dict[tuple, int] = defaultdict(int)
    count_a: dict[object, int] = defaultdict(int)
    count_b: dict[object, int] = defaultdict(int)
    for la, lb in zip(labels_a, labels_b):
        joint[(la, lb)] += 1
        count_a[la] += 1
        count_b[lb] += 1
    mi = 0.0
    for (la, lb), nij in joint.items():
        if nij == 0:
            continue
        pij = nij / n
        pi = count_a[la] / n
        pj = count_b[lb] / n
        mi += pij * log(pij / (pi * pj))
    h_a = -sum((c / n) * log(c / n) for c in count_a.values() if c > 0)
    h_b = -sum((c / n) * log(c / n) for c in count_b.values() if c > 0)
    if h_a + h_b == 0:
        return 1.0
    return 2 * mi / (h_a + h_b)


def compute_ari(labels_a: list[object], labels_b: list[object]) -> float:
    n = len(labels_a)
    assert n == len(labels_b) and n > 0
    if n < 2:
        return 1.0
    joint: dict[tuple, int] = defaultdict(int)
    count_a: dict[object, int] = defaultdict(int)
    count_b: dict[object, int] = defaultdict(int)
    for la, lb in zip(labels_a, labels_b):
        joint[(la, lb)] += 1
        count_a[la] += 1
        count_b[lb] += 1
    total_pairs = comb(n, 2)
    sum_joint = sum(comb(c, 2) for c in joint.values() if c >= 2)
    sum_a = sum(comb(c, 2) for c in count_a.values() if c >= 2)
    sum_b = sum(comb(c, 2) for c in count_b.values() if c >= 2)
    expected = (sum_a * sum_b) / total_pairs if total_pairs else 0.0
    max_index = 0.5 * (sum_a + sum_b)
    denom = max_index - expected
    if denom == 0:
        return 1.0
    return (sum_joint - expected) / denom


def file_module(node_id: str) -> str:
    """Extract the file-level module from a node ID."""
    parts = node_id.split(".")
    file_parts = []
    for p in parts:
        if p and p[0].isupper():
            break
        file_parts.append(p)
    return ".".join(file_parts) if file_parts else parts[0]


def package_of(node_id: str) -> str:
    """Extract the parent package from a node ID.

    flask.app -> flask, flask.json.tag -> flask.json, flask -> flask.
    At module level this gives non-trivial baseline clusters for ARI.
    """
    parts = node_id.rsplit(".", 1)
    return parts[0] if len(parts) > 1 else node_id


def investigate_codebase(name: str, source_path: Path):
    print(f"\n{'='*80}")
    print(f"  {name.upper()}: {source_path}")
    print(f"{'='*80}")

    # 1. Parse
    print(f"\n--- PARSING ---")
    graph = parse_python_project(source_path)
    print(f"Nodes: {graph.node_count}")
    for kind in NodeKind:
        count = sum(1 for n in graph.nodes.values() if n.kind == kind)
        if count:
            print(f"  {kind.value}: {count}")

    print(f"Edges: {graph.edge_count}")
    for kind in EdgeKind:
        edges = graph.edges_by_kind(kind)
        count = len(edges)
        if count:
            # Count unique edges
            unique = len(set((e.source, e.target) for e in edges))
            print(f"  {kind.value}: {count} (unique: {unique})")

    # Call graph density
    functions = [n for n in graph.nodes.values() if n.kind == NodeKind.FUNCTION]
    classes = [n for n in graph.nodes.values() if n.kind == NodeKind.CLASS]
    modules = [n for n in graph.nodes.values() if n.kind == NodeKind.MODULE]
    call_edges = graph.edges_by_kind(EdgeKind.CALLS)
    import_edges = graph.edges_by_kind(EdgeKind.IMPORTS)

    print(f"\nCall graph density: {len(call_edges)}/{len(functions)} = "
          f"{len(call_edges)/max(len(functions),1):.2f} calls/function")
    print(f"Import graph density: {len(import_edges)}/{len(modules)} = "
          f"{len(import_edges)/max(len(modules),1):.2f} imports/module")

    # Show nodes with most call edges
    call_out = defaultdict(int)
    call_in = defaultdict(int)
    for e in call_edges:
        call_out[e.source] += 1
        call_in[e.target] += 1

    print(f"\nTop 10 callers (out-degree):")
    for nid, count in sorted(call_out.items(), key=lambda x: -x[1])[:10]:
        print(f"  {count:3d}  {nid}")

    print(f"\nTop 10 callees (in-degree):")
    for nid, count in sorted(call_in.items(), key=lambda x: -x[1])[:10]:
        print(f"  {count:3d}  {nid}")

    # Nodes with zero call edges
    all_call_nodes = set(call_out.keys()) | set(call_in.keys())
    all_functions = set(n.id for n in functions)
    isolated_functions = all_functions - all_call_nodes
    print(f"\nFunctions with zero call edges: {len(isolated_functions)}/{len(functions)} "
          f"({len(isolated_functions)/max(len(functions),1):.1%})")

    # 2. Run analysis at MODULE level (combined layers)
    for level_name, level in [("MODULE", AnalysisLevel.MODULE), ("SYMBOL", AnalysisLevel.SYMBOL)]:
        print(f"\n--- ANALYSIS ({level_name} level, combined layers) ---")
        config = AnalysisProjectionConfig.for_analysis(
            edge_kind=EdgeKind.CALLS,
            combined=True,
            level=level,
        )
        result = analyze(graph, combined=True, projection_config=config)

        print(f"Analysis graph: {result.graph.node_count} nodes, {result.graph.edge_count} edges")
        if result.coverage:
            print(f"Spectral coverage: {result.coverage.spectral_coverage_ratio:.1%} "
                  f"({result.coverage.spectral_node_count}/{result.coverage.analyzed_node_count})")
            print(f"Components: {result.coverage.component_count} "
                  f"(clusterable: {result.coverage.clusterable_component_count})")
            print(f"Largest component: {result.coverage.largest_component_ratio:.1%}")

        if result.spectral:
            print(f"Fiedler value: {result.spectral.fiedler_value:.6f}")
            print(f"Eigenvalues: {result.spectral.eigenvalues[:8].tolist()}")

        print(f"\nModule detection:")
        print(f"  Package fallback: {result.module_detection.package_fallback}")
        print(f"  Chosen k: {result.module_detection.chosen_k}")
        print(f"  Silhouette: {result.module_detection.silhouette}")
        for mod in result.modules:
            share = mod.size / result.graph.node_count if result.graph.node_count else 0
            print(f"  Module {mod.id}: {mod.size} nodes ({share:.1%}), "
                  f"confidence={mod.confidence:.3f}, "
                  f"unassigned={mod.unassigned}")
            # Show members
            for nid in sorted(mod.node_ids)[:8]:
                print(f"    - {nid}")
            if mod.size > 8:
                print(f"    ... and {mod.size - 8} more")

        # Compare modules against file/package grouping
        if not result.module_detection.package_fallback and result.modules:
            spectral_labels = {}
            file_labels = {}
            pkg_labels = {}
            for mod in result.modules:
                if mod.unassigned:
                    continue
                for nid in mod.node_ids:
                    spectral_labels[nid] = mod.id
                    file_labels[nid] = file_module(nid)
                    pkg_labels[nid] = package_of(nid)

            if len(spectral_labels) > 1:
                common = sorted(spectral_labels.keys())
                sl = [spectral_labels[n] for n in common]
                fl = [file_labels[n] for n in common]
                pl = [pkg_labels[n] for n in common]
                nmi = compute_nmi(sl, fl)
                # ARI uses package-level baseline (file-level gives all
                # singletons at MODULE granularity, making ARI degenerate)
                n_pkg_groups = len(set(pl))
                if n_pkg_groups > 1:
                    ari = compute_ari(sl, pl)
                    print(f"\n  NMI vs file-module baseline: {nmi:.3f}")
                    print(f"  ARI vs package baseline: {ari:.3f}")
                else:
                    print(f"\n  NMI vs file-module baseline: {nmi:.3f}")
                    print(f"  ARI: N/A (single package, no meaningful baseline)")

                # Cross-tab: which file-modules are grouped together?
                module_to_files = defaultdict(lambda: defaultdict(int))
                for nid in common:
                    module_to_files[spectral_labels[nid]][file_labels[nid]] += 1

                print(f"\n  Module composition (spectral module -> file modules):")
                for mid in sorted(module_to_files.keys()):
                    files = module_to_files[mid]
                    total = sum(files.values())
                    parts = sorted(files.items(), key=lambda x: -x[1])
                    print(f"    Module {mid} ({total} nodes): "
                          + ", ".join(f"{f}({c})" for f, c in parts[:6]))

        # Roles
        role_counts: dict[str, int] = defaultdict(int)
        for r in result.roles:
            role_counts[r.role.value] += 1
        print(f"\nRoles:")
        for role, count in sorted(role_counts.items()):
            print(f"  {role}: {count}")

        # Findings
        print(f"\nFindings ({len(result.findings)}):")
        for f in result.findings:
            print(f"  [{f.severity_label}/{f.confidence_label}] {f.kind}: {f.title}")
            print(f"    {f.description}")

        # Anomalies detail
        print(f"\nAnomalies ({len(result.anomalies)}):")
        for a in result.anomalies[:10]:
            print(f"  {a.kind.value}: {a.description} "
                  f"(severity={a.severity:.2f}, confidence={a.confidence:.2f})")

        # Health
        if result.health:
            print(f"\nHealth:")
            print(f"  Call density: {result.health.call_density:.2f}")
            print(f"  Orphans: {result.health.orphan_count}/{result.health.analyzed_node_count} "
                  f"({result.health.orphan_ratio:.1%})")
            print(f"  Largest module: {result.health.largest_module_ratio:.1%} "
                  f"({result.health.largest_module_status})")

    # 3. Also run calls-only analysis to compare
    print(f"\n--- ANALYSIS (MODULE level, calls-only) ---")
    config = AnalysisProjectionConfig.for_analysis(
        edge_kind=EdgeKind.CALLS,
        combined=False,
        level=AnalysisLevel.MODULE,
    )
    result = analyze(graph, edge_kind=EdgeKind.CALLS, projection_config=config)
    print(f"Analysis graph: {result.graph.node_count} nodes, {result.graph.edge_count} edges")
    print(f"Package fallback: {result.module_detection.package_fallback}")
    print(f"Chosen k: {result.module_detection.chosen_k}")
    print(f"Silhouette: {result.module_detection.silhouette}")
    for mod in result.modules:
        share = mod.size / result.graph.node_count if result.graph.node_count else 0
        print(f"  Module {mod.id}: {mod.size} nodes ({share:.1%}), "
              f"confidence={mod.confidence:.3f}")
        for nid in sorted(mod.node_ids)[:5]:
            print(f"    - {nid}")
        if mod.size > 5:
            print(f"    ... and {mod.size - 5} more")


CODEBASES = {
    "Flask": Path("/tmp/flask-source/src/flask"),
    "Requests": Path("/tmp/requests-source/src/requests"),
    "Click": Path("/tmp/click-source/src/click"),
    "FastAPI": Path("/tmp/fastapi-source/fastapi"),
}

if __name__ == "__main__":
    for name, path in CODEBASES.items():
        if not path.exists():
            print(f"SKIP {name}: {path} not found")
            continue
        try:
            investigate_codebase(name, path)
        except Exception as e:
            print(f"ERROR on {name}: {e}")
            import traceback
            traceback.print_exc()
