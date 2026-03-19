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

from topo_parser.graph import CodeGraph, EdgeKind, NodeKind
from topo_analyzer.modules import Module
from topo_analyzer.projection import AnalysisAnchor, AnalysisProjection
from topo_analyzer.spectral import SpectralResult


class AnomalyKind(Enum):
    """Types of structural anomaly."""

    CROSS_MODULE = "cross_module"
    SPECTRAL_OUTLIER = "spectral_outlier"
    CYCLE_MEMBER = "cycle_member"
    LAYER_DISCREPANCY = "layer_discrepancy"


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
    anomalies.extend(_detect_layer_discrepancies(graph, kinds, projection))
    return sorted(anomalies, key=lambda anomaly: (-anomaly.severity, -anomaly.confidence))


def _detect_cross_module(
    graph: CodeGraph,
    modules: list[Module],
    edge_kinds: list[EdgeKind] | EdgeKind,
    projection: AnalysisProjection | None = None,
) -> list[Anomaly]:
    """Detect unexpected cross-module dependencies.

    Detects two patterns:
    1. Bidirectional boundaries: two modules with edges flowing both ways.
    2. Minority couplings: a directed module pair with significantly fewer
       edges than the graph's typical cross-module pair, suggesting a
       layer skip or reverse dependency.
    """
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

    # Compute median edge count per directed pair for minority detection.
    directed_totals = [sum(counts.values()) for counts in pair_counts.values()]
    median_edges = float(np.median(directed_totals)) if directed_totals else 0.0

    for src_mod, tgt_mod in pair_counts:
        module_pair = tuple(sorted((src_mod, tgt_mod)))
        if module_pair in seen_pairs:
            continue
        seen_pairs.add(module_pair)

        forward = pair_counts.get((module_pair[0], module_pair[1]), {})
        reverse = pair_counts.get((module_pair[1], module_pair[0]), {})

        # Pattern 1: Bidirectional boundary (existing detection).
        if forward and reverse:
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
            continue

        # Pattern 2: Minority unidirectional coupling.
        # A directed pair with far fewer edges than typical suggests a
        # layer-skip or reverse dependency — structurally unusual coupling.
        active = forward or reverse
        if not active:
            continue
        active_total = sum(active.values())
        if median_edges <= 0 or active_total >= median_edges:
            continue

        minority_ratio = active_total / median_edges
        # Determine which direction the minority edges flow.
        if forward and not reverse:
            direction_pair = (module_pair[0], module_pair[1])
        else:
            direction_pair = (module_pair[1], module_pair[0])

        examples = pair_examples.get(direction_pair, [])
        anomalies.append(Anomaly(
            kind=AnomalyKind.CROSS_MODULE,
            node_ids=sorted({
                node_id
                for source, target, _ in examples
                for node_id in (source, target)
            }),
            description=(
                f"Unusual dependency from module {direction_pair[0]} to module {direction_pair[1]}: "
                f"{active_total} edges ({minority_ratio:.0%} of typical cross-module coupling)"
            ),
            severity=min(1.0, 0.3 + (1.0 - minority_ratio) * 0.5),
            confidence=min(1.0, 0.4 + active_total / 10.0),
            anchors=_anchors_for_examples(projection, examples),
            edge_counts=dict(active),
        ))
    return anomalies


def _module_label_for_anomaly(module: Module) -> str:
    """Derive a human-readable label from a module's member IDs."""
    if not module.node_ids:
        return f"module_{module.id}"
    prefixes = [nid.split(".")[0] for nid in module.node_ids]
    common = max(set(prefixes), key=prefixes.count)
    return common


def _detect_spectral_outliers(
    spectral: SpectralResult,
    modules: list[Module],
    projection: AnalysisProjection | None,
    threshold_sigma: float = 2.0,
) -> list[Anomaly]:
    """Find nodes whose spectral fingerprint is far from their module centroid."""
    # Precompute centroids for all non-trivial modules.
    module_centroids: dict[int, np.ndarray] = {}
    module_labels: dict[int, str] = {}
    for module in modules:
        if module.unassigned or len(module.node_ids) < 3:
            continue
        fps = []
        for nid in module.node_ids:
            try:
                fps.append(spectral.fingerprint(nid))
            except KeyError:
                continue
        if len(fps) >= 3:
            module_centroids[module.id] = np.vstack(fps).mean(axis=0)
            module_labels[module.id] = _module_label_for_anomaly(module)

    anomalies: list[Anomaly] = []
    for module in modules:
        if module.id not in module_centroids:
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
        centroid = module_centroids[module.id]
        distances = np.linalg.norm(vectors - centroid, axis=1)
        mean_dist = float(distances.mean())
        std_dist = float(distances.std())
        if std_dist == 0:
            continue

        for index, node_id in enumerate(node_ids):
            z_score = (distances[index] - mean_dist) / std_dist
            if z_score <= threshold_sigma:
                continue

            # Find nearest alternative module.
            fp = fingerprints[index]
            nearest_label = ""
            nearest_dist = float("inf")
            for other_id, other_centroid in module_centroids.items():
                if other_id == module.id:
                    continue
                d = float(np.linalg.norm(fp - other_centroid))
                if d < nearest_dist:
                    nearest_dist = d
                    nearest_label = module_labels.get(other_id, str(other_id))

            desc = f"Node {node_id} is {z_score:.1f}σ from module {module.id} centroid"
            if nearest_label:
                desc += f"; nearest alternative: {nearest_label}"

            anomalies.append(Anomaly(
                kind=AnomalyKind.SPECTRAL_OUTLIER,
                node_ids=[node_id],
                description=desc,
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


def _detect_layer_discrepancies(
    graph: CodeGraph,
    edge_kinds: list[EdgeKind],
    projection: AnalysisProjection | None = None,
    gap_threshold: float = 0.4,
) -> list[Anomaly]:
    """Detect nodes with large degree percentile gaps between layers.

    A node that is central in one relationship layer but peripheral in another
    reveals architectural tension — different coupling types disagree about the
    node's structural role.

    Percentiles are computed within each NodeKind group (MODULE, CLASS,
    FUNCTION) rather than globally. This prevents false positives from
    inherent kind-level differences — e.g., MODULE nodes naturally have high
    import degree and zero calls degree in Python.
    """
    if len(edge_kinds) < 2:
        return []

    node_ids = list(graph.nodes)
    if not node_ids:
        return []

    # Group nodes by kind for per-kind percentile computation.
    kind_groups: dict[NodeKind, list[str]] = {}
    for nid in node_ids:
        node = graph.nodes[nid]
        kind_groups.setdefault(node.kind, []).append(nid)

    # Compute degree per layer for each node.
    layer_degrees: dict[EdgeKind, dict[str, int]] = {}
    for kind in edge_kinds:
        degrees: dict[str, int] = {nid: 0 for nid in node_ids}
        for edge in graph.edges_by_kind(kind):
            if edge.source in degrees:
                degrees[edge.source] += 1
            if edge.target in degrees:
                degrees[edge.target] += 1
        layer_degrees[kind] = degrees

    # Compute percentile rank per layer, within each NodeKind group.
    # Skip layers with no edges — all nodes tied at degree 0 is uninformative.
    # Skip kind groups with fewer than 3 nodes — too few for meaningful
    # percentiles.
    layer_percentiles: dict[EdgeKind, dict[str, float]] = {}
    for kind in edge_kinds:
        degrees = layer_degrees[kind]
        if not degrees or max(degrees.values()) == 0:
            continue
        pcts: dict[str, float] = {}
        for _node_kind, group_nids in kind_groups.items():
            if len(group_nids) < 5:
                continue
            group_degrees = sorted(degrees[nid] for nid in group_nids)
            gn = len(group_degrees)
            for nid in group_nids:
                rank = sum(1 for d in group_degrees if d <= degrees[nid])
                pcts[nid] = rank / gn
        layer_percentiles[kind] = pcts

    active_kinds = [k for k in edge_kinds if k in layer_percentiles]
    if len(active_kinds) < 2:
        return []

    anomalies: list[Anomaly] = []
    kind_labels = {
        EdgeKind.CALLS: "calls",
        EdgeKind.IMPORTS: "imports",
        EdgeKind.INHERITS: "inherits",
        EdgeKind.CONTAINS: "contains",
    }

    for nid in node_ids:
        max_deg = max(layer_degrees[k].get(nid, 0) for k in active_kinds)
        if max_deg < 2:
            continue

        # Require meaningful participation in both compared layers — a single
        # edge is too weak to distinguish "peripheral" from "barely present."
        min_deg = min(layer_degrees[k].get(nid, 0) for k in active_kinds)
        if min_deg < 2:
            continue

        # Node must have percentiles in at least 2 layers (may be missing
        # if its kind group was too small).
        node_active = [k for k in active_kinds if nid in layer_percentiles[k]]
        if len(node_active) < 2:
            continue

        best_gap = 0.0
        best_pair: tuple[EdgeKind, EdgeKind] | None = None
        for i, kind_a in enumerate(node_active):
            for kind_b in node_active[i + 1:]:
                pct_a = layer_percentiles[kind_a][nid]
                pct_b = layer_percentiles[kind_b][nid]
                gap = abs(pct_a - pct_b)
                if gap > best_gap:
                    best_gap = gap
                    best_pair = (kind_a, kind_b)

        if best_gap <= gap_threshold or best_pair is None:
            continue

        kind_a, kind_b = best_pair
        pct_a = layer_percentiles[kind_a][nid]
        pct_b = layer_percentiles[kind_b][nid]
        label_a = kind_labels.get(kind_a, kind_a.value)
        label_b = kind_labels.get(kind_b, kind_b.value)

        # Order so the high-percentile layer is described as "central."
        if pct_a >= pct_b:
            desc = f"{nid} is {label_a}-central (p{pct_a:.0%}) but {label_b}-peripheral (p{pct_b:.0%})"
        else:
            desc = f"{nid} is {label_b}-central (p{pct_b:.0%}) but {label_a}-peripheral (p{pct_a:.0%})"

        anomalies.append(Anomaly(
            kind=AnomalyKind.LAYER_DISCREPANCY,
            node_ids=[nid],
            description=desc,
            severity=min(1.0, 0.3 + best_gap * 0.7),
            confidence=0.6,
            anchors=projection.anchors_for([nid], limit=1) if projection else [],
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
