"""Synthetic benchmark validation for clustering quality."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from topo_analyzer.modules import detect_modules
from topo_analyzer.spectral import spectral_decomposition
from topo_parser.graph import CodeGraph, Edge, EdgeKind, Node, NodeKind

from tests.test_validation.benchmark_utils import compute_nmi_from_mappings


def _make_two_cluster_graph(
    n_per_cluster: int = 30,
    intra_p: float = 0.3,
    inter_p: float = 0.02,
    seed: int = 42,
) -> tuple[CodeGraph, dict[str, int]]:
    """Build a synthetic graph with two planted clusters and known ground truth."""
    rng = np.random.default_rng(seed)
    graph = CodeGraph()
    ids_a = [f"cluster_a.fn_{i}" for i in range(n_per_cluster)]
    ids_b = [f"cluster_b.fn_{i}" for i in range(n_per_cluster)]

    for node_id in ids_a + ids_b:
        graph.add_node(Node(
            id=node_id,
            kind=NodeKind.FUNCTION,
            file=Path("/fake"),
            line=1,
            name=node_id,
        ))

    def _add_edges(sources: list[str], targets: list[str], probability: float) -> None:
        for source in sources:
            for target in targets:
                if source != target and rng.random() < probability:
                    graph.add_edge(Edge(source=source, target=target, kind=EdgeKind.CALLS))

    _add_edges(ids_a, ids_a, intra_p)
    _add_edges(ids_b, ids_b, intra_p)
    _add_edges(ids_a, ids_b, inter_p)
    _add_edges(ids_b, ids_a, inter_p)

    return graph, {node_id: 0 for node_id in ids_a} | {node_id: 1 for node_id in ids_b}


def _make_three_cluster_graph(
    n_per_cluster: int = 25,
    intra_p: float = 0.25,
    inter_p: float = 0.01,
    seed: int = 123,
) -> tuple[CodeGraph, dict[str, int]]:
    """Build a synthetic graph with three planted clusters."""
    rng = np.random.default_rng(seed)
    graph = CodeGraph()
    clusters = {
        0: [f"core.fn_{i}" for i in range(n_per_cluster)],
        1: [f"api.fn_{i}" for i in range(n_per_cluster)],
        2: [f"db.fn_{i}" for i in range(n_per_cluster)],
    }
    truth: dict[str, int] = {}
    for cluster_id, node_ids in clusters.items():
        for node_id in node_ids:
            graph.add_node(Node(
                id=node_id,
                kind=NodeKind.FUNCTION,
                file=Path("/fake"),
                line=1,
                name=node_id,
            ))
            truth[node_id] = cluster_id

    all_ids = list(truth.keys())
    for source in all_ids:
        for target in all_ids:
            if source == target:
                continue
            same_cluster = truth[source] == truth[target]
            probability = intra_p if same_cluster else inter_p
            if rng.random() < probability:
                graph.add_edge(Edge(source=source, target=target, kind=EdgeKind.CALLS))

    return graph, truth


def test_fiedler_vector_separates_two_clusters():
    """The Fiedler vector should cleanly separate two planted clusters."""
    graph, truth = _make_two_cluster_graph()
    spectral = spectral_decomposition(graph, EdgeKind.CALLS, k=2)
    assert spectral is not None

    fiedler = spectral.eigenvectors[:, 0]
    predicted = {
        node_id: 0 if fiedler[index] < 0 else 1
        for index, node_id in enumerate(spectral.node_ids)
    }

    common = sorted(set(predicted) & set(truth))
    matches = sum(1 for node_id in common if predicted[node_id] == truth[node_id])
    accuracy = max(matches, len(common) - matches) / len(common)
    assert accuracy > 0.95


def test_detect_modules_recovers_two_clusters():
    """Module detection should recover a two-cluster synthetic graph."""
    graph, truth = _make_two_cluster_graph()
    spectral = spectral_decomposition(graph, EdgeKind.CALLS, k=2)
    assert spectral is not None

    detection = detect_modules(spectral, n_modules=2)
    predicted = {
        node_id: module.id
        for module in detection.modules
        for node_id in module.node_ids
    }
    nmi = compute_nmi_from_mappings(predicted, truth)
    assert nmi > 0.8


def test_auto_k_approximately_recovers_three_clusters():
    """Auto-k should recover roughly the planted three-cluster structure."""
    graph, truth = _make_three_cluster_graph()
    spectral = spectral_decomposition(graph, EdgeKind.CALLS, k=6)
    assert spectral is not None

    detection = detect_modules(spectral)
    predicted = {
        node_id: module.id
        for module in detection.modules
        if not module.unassigned
        for node_id in module.node_ids
    }
    nmi = compute_nmi_from_mappings(predicted, truth)
    assert len(detection.modules) >= 2
    assert detection.chosen_k is not None and detection.chosen_k >= 3
    assert nmi > 0.6
