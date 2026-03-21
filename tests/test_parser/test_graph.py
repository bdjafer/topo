"""Tests for the core graph data model."""

from topo_parser_python.graph import CodeGraph, Edge, EdgeKind, Node, NodeKind


def test_add_node():
    g = CodeGraph()
    node = Node(id="mod.func", kind=NodeKind.FUNCTION, file="mod.py", line=1, name="func")
    g.add_node(node)
    assert g.node_count == 1
    assert "mod.func" in g.nodes


def test_add_edge():
    g = CodeGraph()
    g.add_node(Node(id="a", kind=NodeKind.FUNCTION, file="a.py", line=1, name="a"))
    g.add_node(Node(id="b", kind=NodeKind.FUNCTION, file="b.py", line=1, name="b"))
    g.add_edge(Edge(source="a", target="b", kind=EdgeKind.CALLS))
    assert g.edge_count == 1


def test_edges_by_kind():
    g = CodeGraph()
    g.add_edge(Edge(source="a", target="b", kind=EdgeKind.CALLS))
    g.add_edge(Edge(source="a", target="c", kind=EdgeKind.IMPORTS))
    g.add_edge(Edge(source="b", target="c", kind=EdgeKind.CALLS))
    assert len(g.edges_by_kind(EdgeKind.CALLS)) == 2
    assert len(g.edges_by_kind(EdgeKind.IMPORTS)) == 1


def test_summary():
    g = CodeGraph()
    g.add_node(Node(id="mod", kind=NodeKind.MODULE, file="mod.py", line=1, name="mod"))
    g.add_node(Node(id="mod.f", kind=NodeKind.FUNCTION, file="mod.py", line=5, name="f"))
    g.add_edge(Edge(source="mod", target="mod.f", kind=EdgeKind.CONTAINS))
    summary = g.summary()
    assert "2 nodes" in summary
    assert "1 edges" in summary
