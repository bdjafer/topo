"""
Structural anomaly detection.

Identifies structural findings that are likely to matter to developers:
unexpected bidirectional boundaries, cluster-local spectral outliers, and
strongly connected dependency groups.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import networkx as nx
import numpy as np

from topo_parser.graph import CodeGraph, EdgeKind
from topo_analyzer.modules import Module
from topo_analyzer.projection import AnalysisAnchor, AnalysisProjection
from topo_analyzer.spectral import SpectralResult


class AnomalyKind(Enum):
    """Types of structural anomaly."""

    CROSS_MODULE = "cross_module"
    SPECTRAL_OUTLIER = "spectral_outlier"
    CYCLE_MEMBER = "cycle_member"


@dataclass
class Anomaly:
    """A detected structural anomaly."""

    kind: AnomalyKind
    node_ids: list[str]
    description: str
    severity: float
    confidence: float = 0.5
    anchors: list[AnalysisAnchor] = field(default_factory=list)
    edge_counts: dict[EdgeKind, int] = field(default_factory=dict)


def detect_anomalies(
    graph: CodeGraph,
    spectral: SpectralResult | None,
    modules: list[Module],
    edge_kind: EdgeKind = EdgeKind.CALLS,
    edge_kinds: list[EdgeKind] | None = None,
    projection: AnalysisProjection | None = None,
) -> list[Anomaly]:
    """Run all anomaly detectors and return combined results."""
    kinds = edge_kinds if edge_kinds is not None else [edge_kind]
    anomalies: list[Anomaly] = []
    anomalies.extend(_detect_cross_module(graph, modules, kinds, projection))
    if spectral and modules:
        anomalies.extend(_detect_spectral_outliers(spectral, modules, projection))
    anomalies.extend(_detect_cycles(graph, kinds, projection))
    return sorted(anomalies, key=lambda anomaly: (-anomaly.severity, -anomaly.confidence))


def _detect_cross_module(
    graph: CodeGraph,
    modules: list[Module],
    edge_kinds: list[EdgeKind] | EdgeKind,
    projection: AnalysisProjection | None = None,
) -> list[Anomaly]:
    """Group unexpected bidirectional boundaries between detected modules."""
    if not modules:
        return []
    kinds = edge_kinds if isinstance(edge_kinds, list) else [edge_kinds]

    node_to_module: dict[str, int] = {}
    unassigned_modules: set[int] = set()
    for module in modules:
        for node_id in module.node_ids:
            node_to_module[node_id] = module.id
        if module.unassigned:
            unassigned_modules.add(module.id)

    pair_counts: dict[tuple[int, int], dict[EdgeKind, int]] = {}
    pair_examples: dict[tuple[int, int], list[tuple[str, str, EdgeKind]]] = {}
    for edge_kind in kinds:
        for edge in graph.edges_by_kind(edge_kind):
            src_mod = node_to_module.get(edge.source)
            tgt_mod = node_to_module.get(edge.target)
            if src_mod is None or tgt_mod is None or src_mod == tgt_mod:
                continue
            if src_mod in unassigned_modules or tgt_mod in unassigned_modules:
                continue
            pair = (src_mod, tgt_mod)
            counts = pair_counts.setdefault(pair, {})
            counts[edge_kind] = counts.get(edge_kind, 0) + 1
            pair_examples.setdefault(pair, [])
            if len(pair_examples[pair]) < 3:
                pair_examples[pair].append((edge.source, edge.target, edge_kind))

    anomalies: list[Anomaly] = []
    seen_pairs: set[tuple[int, int]] = set()
    for src_mod, tgt_mod in pair_counts:
        module_pair = tuple(sorted((src_mod, tgt_mod)))
        if module_pair in seen_pairs:
            continue
        seen_pairs.add(module_pair)

        forward = pair_counts.get((module_pair[0], module_pair[1]), {})
        reverse = pair_counts.get((module_pair[1], module_pair[0]), {})
        if not forward or not reverse:
            continue

        forward_total = sum(forward.values())
        reverse_total = sum(reverse.values())
        total_edges = forward_total + reverse_total
        reverse_share = min(forward_total, reverse_total) / total_edges
        edge_counts = dict(forward)
        for edge_kind, count in reverse.items():
            edge_counts[edge_kind] = edge_counts.get(edge_kind, 0) + count

        examples = pair_examples.get((module_pair[0], module_pair[1]), []) + pair_examples.get((module_pair[1], module_pair[0]), [])
        anomalies.append(Anomaly(
            kind=AnomalyKind.CROSS_MODULE,
            node_ids=sorted({
                node_id
                for source, target, _ in examples
                for node_id in (source, target)
            }),
            description=(
                f"Bidirectional dependency between module {module_pair[0]} and module {module_pair[1]}: "
                f"{forward_total} forward edges, {reverse_total} reverse edges"
            ),
            severity=min(1.0, 0.35 + reverse_share),
            confidence=min(1.0, 0.5 + total_edges / 20.0),
            anchors=_anchors_for_examples(projection, examples),
            edge_counts=edge_counts,
        ))
    return anomalies


def _detect_spectral_outliers(
    spectral: SpectralResult,
    modules: list[Module],
    projection: AnalysisProjection | None,
    threshold_sigma: float = 2.0,
) -> list[Anomaly]:
    """Find nodes whose spectral fingerprint is far from their module centroid."""
    anomalies: list[Anomaly] = []
    for module in modules:
        if module.unassigned or len(module.node_ids) < 3:
            continue

        fingerprints: list[np.ndarray] = []
        node_ids: list[str] = []
        for node_id in module.node_ids:
            try:
                fingerprints.append(spectral.fingerprint(node_id))
                node_ids.append(node_id)
            except KeyError:
                continue
        if len(fingerprints) < 3:
            continue

        vectors = np.vstack(fingerprints)
        centroid = vectors.mean(axis=0)
        distances = np.linalg.norm(vectors - centroid, axis=1)
        mean_dist = float(distances.mean())
        std_dist = float(distances.std())
        if std_dist == 0:
            continue

        for index, node_id in enumerate(node_ids):
            z_score = (distances[index] - mean_dist) / std_dist
            if z_score <= threshold_sigma:
                continue
            anomalies.append(Anomaly(
                kind=AnomalyKind.SPECTRAL_OUTLIER,
                node_ids=[node_id],
                description=f"Node {node_id} is {z_score:.1f}σ from module {module.id} centroid",
                severity=min(1.0, z_score / 4.0),
                confidence=max(0.4, module.confidence),
                anchors=projection.anchors_for([node_id], limit=1) if projection else [],
            ))
    return anomalies


def _detect_cycles(
    graph: CodeGraph,
    edge_kinds: list[EdgeKind] | EdgeKind,
    projection: AnalysisProjection | None = None,
) -> list[Anomaly]:
    """Find strongly connected dependency groups."""
    nx_graph = nx.DiGraph()
    kinds = edge_kinds if isinstance(edge_kinds, list) else [edge_kinds]
    for edge_kind in kinds:
        for edge in graph.edges_by_kind(edge_kind):
            if edge.source in graph.nodes and edge.target in graph.nodes:
                nx_graph.add_edge(edge.source, edge.target)

    anomalies: list[Anomaly] = []
    try:
        components = list(nx.strongly_connected_components(nx_graph))
    except nx.NetworkXError:
        return []

    for component in components:
        if len(component) <= 1:
            continue
        node_ids = sorted(component)
        anomalies.append(Anomaly(
            kind=AnomalyKind.CYCLE_MEMBER,
            node_ids=node_ids,
            description=f"Strongly connected dependency group of {len(node_ids)} nodes",
            severity=min(1.0, len(node_ids) / 8.0),
            confidence=min(1.0, 0.4 + len(node_ids) / 10.0),
            anchors=projection.anchors_for(node_ids) if projection else [],
        ))
    return anomalies


def _anchors_for_examples(
    projection: AnalysisProjection | None,
    examples: list[tuple[str, str, EdgeKind]],
) -> list[AnalysisAnchor]:
    """Collect a short list of anchors for representative anomalous edges."""
    if projection is None:
        return []
    node_ids: list[str] = []
    for source, target, _ in examples:
        node_ids.extend([source, target])
    return projection.anchors_for(node_ids)
