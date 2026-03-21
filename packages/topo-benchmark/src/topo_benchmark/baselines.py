"""Simple baselines for benchmark comparison."""

from __future__ import annotations

import networkx as nx

from topo_parser_python.graph import CodeGraph, EdgeKind


def directory_partition(graph: CodeGraph) -> dict[str, str]:
    """Group nodes by their first dotted path component (top-level package)."""
    return {node_id: node_id.split(".", 1)[0] for node_id in graph.nodes}


def directory_partition_by_module(graph: CodeGraph, module_ids: set[str]) -> dict[str, str]:
    """Group symbol-level nodes by their owning module's second dotted component.

    For symbol nodes like ``click.core.Command.__init__``, find the longest
    prefix that matches a known module ID (e.g. ``click.core``), then use
    that module's second component (``click.core``) as the directory label.
    Falls back to first component if there is no second.
    """
    partition: dict[str, str] = {}
    sorted_modules = sorted(module_ids, key=len, reverse=True)
    for node_id in graph.nodes:
        owning_module = node_id
        for mod in sorted_modules:
            if node_id == mod or node_id.startswith(mod + "."):
                owning_module = mod
                break
        parts = owning_module.split(".")
        label = ".".join(parts[:2]) if len(parts) >= 2 else parts[0]
        partition[node_id] = label
    return partition


def louvain_partition(
    graph: CodeGraph,
    edge_kinds: list[EdgeKind] | None = None,
    seed: int = 42,
) -> dict[str, int]:
    """Louvain community detection on the projected graph."""
    if edge_kinds is None:
        edge_kinds = [EdgeKind.CALLS, EdgeKind.IMPORTS, EdgeKind.INHERITS]

    G = nx.Graph()
    G.add_nodes_from(graph.nodes)
    for edge in graph.edges:
        if edge.kind in edge_kinds:
            if G.has_edge(edge.source, edge.target):
                G[edge.source][edge.target]["weight"] += 1
            else:
                G.add_edge(edge.source, edge.target, weight=1)

    communities = nx.community.louvain_communities(G, seed=seed)
    partition: dict[str, int] = {}
    for i, community in enumerate(communities):
        for node_id in community:
            partition[node_id] = i
    return partition


def heuristic_anomalies(
    graph: CodeGraph,
    edge_kinds: list[EdgeKind] | None = None,
) -> list[dict]:
    """Simple anomaly detection: SCC-based cycles, cross-module edge counts."""
    if edge_kinds is None:
        edge_kinds = [EdgeKind.CALLS, EdgeKind.IMPORTS, EdgeKind.INHERITS]

    G = nx.DiGraph()
    G.add_nodes_from(graph.nodes)
    for edge in graph.edges:
        if edge.kind in edge_kinds:
            G.add_edge(edge.source, edge.target)

    dir_part = directory_partition(graph)
    anomalies: list[dict] = []

    # SCC-based cycle detection
    for scc in nx.strongly_connected_components(G):
        if len(scc) > 1:
            anomalies.append({
                "kind": "cycle_member",
                "node_ids": sorted(scc),
                "severity": min(1.0, len(scc) / 10),
                "confidence": 0.8,
            })

    # Cross-module edge counting per module pair
    cross_counts: dict[tuple[str, str], int] = {}
    for edge in graph.edges:
        if edge.kind not in edge_kinds:
            continue
        src_mod = dir_part.get(edge.source, "")
        tgt_mod = dir_part.get(edge.target, "")
        if src_mod and tgt_mod and src_mod != tgt_mod:
            key = (src_mod, tgt_mod)
            cross_counts[key] = cross_counts.get(key, 0) + 1

    for (src_mod, tgt_mod), count in sorted(
        cross_counts.items(), key=lambda x: -x[1]
    ):
        anomalies.append({
            "kind": "cross_module",
            "node_ids": [src_mod, tgt_mod],
            "severity": min(1.0, count / 5),
            "confidence": 0.6,
        })

    anomalies.sort(key=lambda a: -a["severity"])
    return anomalies
