"""Tests for disconnected-graph spectral handling."""

from pathlib import Path

from topo_analyzer.spectral import spectral_decomposition
from topo_parser.graph import CodeGraph, Edge, EdgeKind, Node, NodeKind


def test_disconnected_spectral_marks_small_components_unassigned():
    """Disconnected nodes should no longer collapse into the main embedding."""
    graph = CodeGraph()
    for name in ["a0", "a1", "a2", "b0", "b1"]:
        graph.add_node(Node(id=name, kind=NodeKind.FUNCTION, file=Path("m.py"), line=1, name=name))
    graph.add_edge(Edge(source="a0", target="a1", kind=EdgeKind.CALLS))
    graph.add_edge(Edge(source="a1", target="a2", kind=EdgeKind.CALLS))
    graph.add_edge(Edge(source="b0", target="b1", kind=EdgeKind.CALLS))

    result = spectral_decomposition(graph, edge_kind=EdgeKind.CALLS)

    assert result is not None
    assert result.node_ids == ["a0", "a1", "a2"]
    assert result.unassigned_node_ids == ["b0", "b1"]
    assert result.coverage_ratio == 3 / 5
