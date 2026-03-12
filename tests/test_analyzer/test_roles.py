"""Tests for structural role classification."""

from pathlib import Path

from topo_parser.graph import CodeGraph, Edge, EdgeKind, Node, NodeKind
from topo_analyzer.roles import StructuralRole, classify_roles


def test_orphan_detection():
    """A node with no edges should be classified as orphan."""
    g = CodeGraph()
    g.add_node(Node(id="lonely", kind=NodeKind.FUNCTION, file=Path("m.py"), line=1, name="lonely"))
    roles = classify_roles(g, edge_kind=EdgeKind.CALLS)
    assert len(roles) == 1
    assert roles[0].role == StructuralRole.ORPHAN


def test_hub_detection():
    """A node called by many others should be classified as a hub."""
    g = CodeGraph()
    g.add_node(Node(id="hub", kind=NodeKind.FUNCTION, file=Path("m.py"), line=1, name="hub"))
    for i in range(10):
        nid = f"caller{i}"
        g.add_node(Node(id=nid, kind=NodeKind.FUNCTION, file=Path("m.py"), line=i+2, name=nid))
        g.add_edge(Edge(source=nid, target="hub", kind=EdgeKind.CALLS))
        # Hub also calls back (to boost total degree)
        g.add_edge(Edge(source="hub", target=nid, kind=EdgeKind.CALLS))

    roles = classify_roles(g, edge_kind=EdgeKind.CALLS)
    hub_role = next(r for r in roles if r.node_id == "hub")
    assert hub_role.role == StructuralRole.HUB
