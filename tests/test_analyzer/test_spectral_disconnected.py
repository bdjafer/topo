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


def test_fingerprint_width_consistent_across_components():
    """Fingerprints from different-sized components should have uniform width."""
    graph = CodeGraph()
    # Component 1: 5 nodes (will get up to k=3 eigenvectors)
    for i in range(5):
        graph.add_node(Node(id=f"big.{i}", kind=NodeKind.FUNCTION, file=Path("m.py"), line=1, name=f"b{i}"))
    for i in range(4):
        graph.add_edge(Edge(source=f"big.{i}", target=f"big.{i+1}", kind=EdgeKind.CALLS))
    # Component 2: 3 nodes (will get k=1 eigenvector)
    for i in range(3):
        graph.add_node(Node(id=f"sm.{i}", kind=NodeKind.FUNCTION, file=Path("m.py"), line=1, name=f"s{i}"))
    graph.add_edge(Edge(source="sm.0", target="sm.1", kind=EdgeKind.CALLS))
    graph.add_edge(Edge(source="sm.1", target="sm.2", kind=EdgeKind.CALLS))

    result = spectral_decomposition(graph, edge_kind=EdgeKind.CALLS)
    assert result is not None
    assert len(result.components) == 2

    # All fingerprints should match the eigenvectors matrix width
    expected_width = result.eigenvectors.shape[1]
    for node_id in result.node_ids:
        fp = result.fingerprint(node_id)
        assert fp.shape == (expected_width,), (
            f"{node_id}: fingerprint width {fp.shape[0]} != eigenvectors width {expected_width}"
        )
