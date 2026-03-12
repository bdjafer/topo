"""
Structural anomaly detection.

Identifies code entities that are structurally unusual: unexpected
cross-module dependencies, spectral outliers, and dependency cycles.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import networkx as nx
import numpy as np

from topo_parser.graph import CodeGraph, EdgeKind
from topo_analyzer.spectral import SpectralResult
from topo_analyzer.modules import Module


class AnomalyKind(Enum):
    """Types of structural anomaly."""

    CROSS_MODULE = "cross_module"  # Edge between nodes in different detected modules
    SPECTRAL_OUTLIER = "spectral_outlier"  # Node far from its cluster centroid
    CYCLE_MEMBER = "cycle_member"  # Node participates in a dependency cycle


@dataclass
class Anomaly:
    """A detected structural anomaly."""

    kind: AnomalyKind
    node_ids: list[str]  # Nodes involved
    description: str
    severity: float  # 0.0–1.0, higher = more anomalous


def detect_anomalies(
    graph: CodeGraph,
    spectral: SpectralResult | None,
    modules: list[Module],
    edge_kind: EdgeKind = EdgeKind.CALLS,
) -> list[Anomaly]:
    """Run all anomaly detectors and return combined results."""
    anomalies: list[Anomaly] = []
    anomalies.extend(_detect_cross_module(graph, modules, edge_kind))
    if spectral and modules:
        anomalies.extend(_detect_spectral_outliers(spectral, modules))
    anomalies.extend(_detect_cycles(graph, edge_kind))
    return sorted(anomalies, key=lambda a: -a.severity)


def _detect_cross_module(
    graph: CodeGraph,
    modules: list[Module],
    edge_kind: EdgeKind,
) -> list[Anomaly]:
    """Find edges that cross detected module boundaries."""
    if not modules:
        return []

    # Map node -> module id
    node_to_module: dict[str, int] = {}
    for mod in modules:
        for nid in mod.node_ids:
            node_to_module[nid] = mod.id

    anomalies: list[Anomaly] = []
    seen: set[tuple[str, str]] = set()
    for edge in graph.edges_by_kind(edge_kind):
        src_mod = node_to_module.get(edge.source)
        tgt_mod = node_to_module.get(edge.target)
        if src_mod is not None and tgt_mod is not None and src_mod != tgt_mod:
            pair = (edge.source, edge.target)
            if pair not in seen:
                seen.add(pair)
                anomalies.append(Anomaly(
                    kind=AnomalyKind.CROSS_MODULE,
                    node_ids=[edge.source, edge.target],
                    description=f"Edge from module {src_mod} to module {tgt_mod}: {edge.source} -> {edge.target}",
                    severity=0.5,
                ))
    return anomalies


def _detect_spectral_outliers(
    spectral: SpectralResult,
    modules: list[Module],
    threshold_sigma: float = 2.0,
) -> list[Anomaly]:
    """Find nodes whose spectral fingerprint is far from their cluster centroid."""
    # Build module membership
    node_to_module: dict[str, int] = {}
    for mod in modules:
        for nid in mod.node_ids:
            node_to_module[nid] = mod.id

    # Compute cluster centroids
    centroids: dict[int, np.ndarray] = {}
    for mod in modules:
        indices = [
            spectral.node_ids.index(nid)
            for nid in mod.node_ids
            if nid in spectral.node_ids
        ]
        if indices:
            centroids[mod.id] = spectral.eigenvectors[indices].mean(axis=0)

    # Compute distances from centroid
    distances: list[tuple[str, float, int]] = []
    for nid in spectral.node_ids:
        mid = node_to_module.get(nid)
        if mid is not None and mid in centroids:
            idx = spectral.node_ids.index(nid)
            fp = spectral.eigenvectors[idx]
            dist = float(np.linalg.norm(fp - centroids[mid]))
            distances.append((nid, dist, mid))

    if not distances:
        return []

    all_dists = np.array([d[1] for d in distances])
    mean_dist = float(all_dists.mean())
    std_dist = float(all_dists.std())
    if std_dist == 0:
        return []

    anomalies: list[Anomaly] = []
    for nid, dist, mid in distances:
        z_score = (dist - mean_dist) / std_dist
        if z_score > threshold_sigma:
            severity = min(1.0, z_score / 5.0)
            anomalies.append(Anomaly(
                kind=AnomalyKind.SPECTRAL_OUTLIER,
                node_ids=[nid],
                description=f"Node {nid} is {z_score:.1f}σ from module {mid} centroid",
                severity=severity,
            ))
    return anomalies


def _detect_cycles(
    graph: CodeGraph,
    edge_kind: EdgeKind,
) -> list[Anomaly]:
    """Find nodes that participate in dependency cycles."""
    nx_graph = nx.DiGraph()
    for edge in graph.edges_by_kind(edge_kind):
        if edge.source in graph.nodes and edge.target in graph.nodes:
            nx_graph.add_edge(edge.source, edge.target)

    anomalies: list[Anomaly] = []
    try:
        cycles = list(nx.simple_cycles(nx_graph))
    except nx.NetworkXError:
        return []

    for cycle in cycles:
        severity = min(1.0, len(cycle) / 10.0)
        cycle_str = " -> ".join(cycle) + " -> " + cycle[0]
        anomalies.append(Anomaly(
            kind=AnomalyKind.CYCLE_MEMBER,
            node_ids=list(cycle),
            description=f"Dependency cycle: {cycle_str}",
            severity=severity,
        ))

    return anomalies
