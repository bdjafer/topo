"""Tests for structural anomaly detection."""

from pathlib import Path

from topo_parser.graph import CodeGraph, Edge, EdgeKind, Node, NodeKind
from topo_analyzer.anomalies import (
    AnomalyKind,
    detect_anomalies,
    _detect_cross_module,
    _detect_cycles,
)
from topo_analyzer.modules import Module


def test_cross_module_anomaly():
    """An edge crossing module boundaries should be flagged."""
    g = CodeGraph()
    g.add_node(Node(id="a", kind=NodeKind.FUNCTION, file=Path("a.py"), line=1, name="a"))
    g.add_node(Node(id="b", kind=NodeKind.FUNCTION, file=Path("b.py"), line=1, name="b"))
    g.add_edge(Edge(source="a", target="b", kind=EdgeKind.CALLS))

    modules = [
        Module(id=0, node_ids=["a"]),
        Module(id=1, node_ids=["b"]),
    ]

    anomalies = _detect_cross_module(g, modules, EdgeKind.CALLS)
    assert len(anomalies) == 1
    assert anomalies[0].kind == AnomalyKind.CROSS_MODULE
    assert "a" in anomalies[0].node_ids
    assert "b" in anomalies[0].node_ids


def test_no_cross_module_within_same_module():
    """An edge within the same module should not be flagged."""
    g = CodeGraph()
    g.add_node(Node(id="a", kind=NodeKind.FUNCTION, file=Path("a.py"), line=1, name="a"))
    g.add_node(Node(id="b", kind=NodeKind.FUNCTION, file=Path("b.py"), line=1, name="b"))
    g.add_edge(Edge(source="a", target="b", kind=EdgeKind.CALLS))

    modules = [Module(id=0, node_ids=["a", "b"])]
    anomalies = _detect_cross_module(g, modules, EdgeKind.CALLS)
    assert len(anomalies) == 0


def test_cycle_detection():
    """A dependency cycle should be detected."""
    g = CodeGraph()
    for name in ["a", "b", "c"]:
        g.add_node(Node(id=name, kind=NodeKind.FUNCTION, file=Path("m.py"), line=1, name=name))
    g.add_edge(Edge(source="a", target="b", kind=EdgeKind.CALLS))
    g.add_edge(Edge(source="b", target="c", kind=EdgeKind.CALLS))
    g.add_edge(Edge(source="c", target="a", kind=EdgeKind.CALLS))

    anomalies = _detect_cycles(g, EdgeKind.CALLS)
    assert len(anomalies) >= 1
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
