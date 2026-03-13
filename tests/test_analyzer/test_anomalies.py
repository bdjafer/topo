"""Tests for structural anomaly detection."""

from pathlib import Path

import numpy as np

from topo_parser.graph import CodeGraph, Edge, EdgeKind, Node, NodeKind
from topo_analyzer.anomalies import (
    AnomalyKind,
    detect_anomalies,
    _detect_cross_module,
    _detect_spectral_outliers,
    _detect_cycles,
)
from topo_analyzer.modules import Module
from topo_analyzer.spectral import SpectralComponent, SpectralResult


def test_cross_module_anomaly():
    """A bidirectional cross-module dependency should be flagged."""
    g = CodeGraph()
    g.add_node(Node(id="a", kind=NodeKind.FUNCTION, file=Path("a.py"), line=1, name="a"))
    g.add_node(Node(id="b", kind=NodeKind.FUNCTION, file=Path("b.py"), line=1, name="b"))
    g.add_edge(Edge(source="a", target="b", kind=EdgeKind.CALLS))
    g.add_edge(Edge(source="b", target="a", kind=EdgeKind.CALLS))

    modules = [
        Module(id=0, node_ids=["a"]),
        Module(id=1, node_ids=["b"]),
    ]

    anomalies = _detect_cross_module(g, modules, EdgeKind.CALLS)
    assert len(anomalies) == 1
    assert anomalies[0].kind == AnomalyKind.CROSS_MODULE
    assert anomalies[0].edge_counts[EdgeKind.CALLS] == 2


def test_no_cross_module_within_same_module():
    """An edge within the same module should not be flagged."""
    g = CodeGraph()
    g.add_node(Node(id="a", kind=NodeKind.FUNCTION, file=Path("a.py"), line=1, name="a"))
    g.add_node(Node(id="b", kind=NodeKind.FUNCTION, file=Path("b.py"), line=1, name="b"))
    g.add_edge(Edge(source="a", target="b", kind=EdgeKind.CALLS))

    modules = [Module(id=0, node_ids=["a", "b"])]
    anomalies = _detect_cross_module(g, modules, EdgeKind.CALLS)
    assert len(anomalies) == 0


def test_cross_module_severity_penalizes_reverse_direction():
    """More balanced reverse flow should score as more severe."""
    g = CodeGraph()
    for name in ["a0", "a1", "a2", "b0", "b1", "c0", "c1", "d0", "d1"]:
        g.add_node(Node(id=name, kind=NodeKind.FUNCTION, file=Path("m.py"), line=1, name=name))

    for source, target in [
        ("a0", "b0"), ("a1", "b0"), ("a2", "b1"), ("b1", "a0"),
        ("c0", "d0"), ("d0", "c0"), ("c1", "d1"), ("d1", "c1"),
    ]:
        g.add_edge(Edge(source=source, target=target, kind=EdgeKind.CALLS))

    modules = [
        Module(id=0, node_ids=["a0", "a1", "a2"]),
        Module(id=1, node_ids=["b0", "b1"]),
        Module(id=2, node_ids=["c0", "c1"]),
        Module(id=3, node_ids=["d0", "d1"]),
    ]

    anomalies = _detect_cross_module(g, modules, EdgeKind.CALLS)
    severity_by_pair = {a.description.split(":")[0]: a.severity for a in anomalies}

    assert severity_by_pair["Bidirectional dependency between module 2 and module 3"] > (
        severity_by_pair["Bidirectional dependency between module 0 and module 1"]
    )


def test_cycle_detection():
    """A strongly connected dependency group should be detected."""
    g = CodeGraph()
    for name in ["a", "b", "c"]:
        g.add_node(Node(id=name, kind=NodeKind.FUNCTION, file=Path("m.py"), line=1, name=name))
    g.add_edge(Edge(source="a", target="b", kind=EdgeKind.CALLS))
    g.add_edge(Edge(source="b", target="c", kind=EdgeKind.CALLS))
    g.add_edge(Edge(source="c", target="a", kind=EdgeKind.CALLS))

    anomalies = _detect_cycles(g, EdgeKind.CALLS)
    assert len(anomalies) == 1
    cycle_anomaly = anomalies[0]
    assert cycle_anomaly.kind == AnomalyKind.CYCLE_MEMBER
    assert set(cycle_anomaly.node_ids) == {"a", "b", "c"}


def test_no_cycles_in_dag():
    """A DAG should produce no cycle anomalies."""
    g = CodeGraph()
    for name in ["a", "b", "c"]:
        g.add_node(Node(id=name, kind=NodeKind.FUNCTION, file=Path("m.py"), line=1, name=name))
    g.add_edge(Edge(source="a", target="b", kind=EdgeKind.CALLS))
    g.add_edge(Edge(source="b", target="c", kind=EdgeKind.CALLS))

    anomalies = _detect_cycles(g, EdgeKind.CALLS)
    assert len(anomalies) == 0


def test_detect_anomalies_integration():
    """Full anomaly detection pipeline runs without errors."""
    g = CodeGraph()
    g.add_node(Node(id="a", kind=NodeKind.FUNCTION, file=Path("m.py"), line=1, name="a"))
    g.add_node(Node(id="b", kind=NodeKind.FUNCTION, file=Path("m.py"), line=2, name="b"))
    g.add_edge(Edge(source="a", target="b", kind=EdgeKind.CALLS))

    anomalies = detect_anomalies(g, spectral=None, modules=[], edge_kind=EdgeKind.CALLS)
    assert isinstance(anomalies, list)


def test_spectral_outlier_detected_within_module():
    """Spectral outliers should be scored relative to their own module."""
    spectral = SpectralResult(
        total_node_ids=["a", "b", "c", "d"],
        components=[
            SpectralComponent(
                id=0,
                node_ids=["a", "b", "c", "d"],
                eigenvalues=np.array([0.1, 0.2]),
                eigenvectors=np.array([
                    [0.0, 0.0],
                    [0.1, 0.0],
                    [0.0, 0.1],
                    [3.0, 3.0],
                ]),
            )
        ],
        unassigned_components=[],
        primary_eigenvalues=np.array([0.1, 0.2]),
        component_sizes=[4],
        fiedler_value=0.1,
    )
    modules = [Module(id=0, node_ids=["a", "b", "c", "d"], confidence=0.8)]

    anomalies = _detect_spectral_outliers(spectral, modules, projection=None, threshold_sigma=1.5)
    assert len(anomalies) == 1
    assert anomalies[0].kind == AnomalyKind.SPECTRAL_OUTLIER
    assert anomalies[0].node_ids == ["d"]


def test_spectral_outlier_multi_component_no_crash():
    """Outlier detection should handle components with different fingerprint widths."""
    spectral = SpectralResult(
        total_node_ids=["a0", "a1", "a2", "a3", "b0", "b1", "b2"],
        components=[
            SpectralComponent(
                id=0,
                node_ids=["a0", "a1", "a2", "a3"],
                eigenvalues=np.array([0.1, 0.2, 0.3]),
                eigenvectors=np.array([
                    [0.0, 0.0, 0.0],
                    [0.1, 0.0, 0.0],
                    [0.0, 0.1, 0.0],
                    [3.0, 3.0, 3.0],
                ]),
            ),
            SpectralComponent(
                id=1,
                node_ids=["b0", "b1", "b2"],
                eigenvalues=np.array([0.5]),
                eigenvectors=np.array([
                    [0.0],
                    [0.1],
                    [5.0],
                ]),
            ),
        ],
        unassigned_components=[],
        primary_eigenvalues=np.array([0.1, 0.2, 0.3]),
        component_sizes=[4, 3],
        fiedler_value=0.1,
    )
    modules = [
        Module(id=0, node_ids=["a0", "a1", "a2", "a3"], confidence=0.8, component_id=0),
        Module(id=1, node_ids=["b0", "b1", "b2"], confidence=0.7, component_id=1),
    ]

    # Should not crash despite different eigenvector widths (3 vs 1)
    anomalies = _detect_spectral_outliers(spectral, modules, projection=None, threshold_sigma=1.5)
    assert isinstance(anomalies, list)
    outlier_nodes = {nid for a in anomalies for nid in a.node_ids}
    assert "a3" in outlier_nodes or "b2" in outlier_nodes
