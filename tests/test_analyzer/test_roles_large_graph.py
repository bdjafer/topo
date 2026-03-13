"""Tests for the approximate large-graph role path."""

from pathlib import Path

import topo_analyzer.roles as roles_module
from topo_analyzer.roles import classify_roles
from topo_parser.graph import CodeGraph, Edge, EdgeKind, Node, NodeKind


def test_large_graph_uses_approximate_betweenness(monkeypatch):
    """The large-graph branch should return sane role assignments."""
    monkeypatch.setattr(roles_module, "_BETWEENNESS_APPROX_THRESHOLD", 5)

    graph = CodeGraph()
    for index in range(20):
        graph.add_node(Node(
            id=f"n{index}",
            kind=NodeKind.FUNCTION,
            file=Path("m.py"),
            line=index,
            name=f"n{index}",
        ))
    for index in range(19):
        graph.add_edge(Edge(source=f"n{index}", target=f"n{index + 1}", kind=EdgeKind.CALLS))

    assignments = classify_roles(graph, edge_kind=EdgeKind.CALLS)

    assert len(assignments) == 20
    assert any(assignment.betweenness > 0 for assignment in assignments)
