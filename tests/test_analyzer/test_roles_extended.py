"""Extended tests for structural role classification."""

from topo_parser.graph import CodeGraph, Edge, EdgeKind, Node, NodeKind
from topo_analyzer.roles import StructuralRole, classify_roles


def test_bridge_detection():
    """A node with high betweenness connecting two clusters should be a bridge."""
    g = CodeGraph()
    # Two clusters connected only through "bridge"
    g.add_node(Node(id="bridge", kind=NodeKind.FUNCTION, file="m.py", line=1, name="bridge"))
    for i in range(4):
        nid = f"left{i}"
        g.add_node(Node(id=nid, kind=NodeKind.FUNCTION, file="m.py", line=i+2, name=nid))
        g.add_edge(Edge(source=nid, target="bridge", kind=EdgeKind.CALLS))
    for i in range(4):
        nid = f"right{i}"
        g.add_node(Node(id=nid, kind=NodeKind.FUNCTION, file="m.py", line=i+10, name=nid))
        g.add_edge(Edge(source="bridge", target=nid, kind=EdgeKind.CALLS))

    roles = classify_roles(g, edge_kind=EdgeKind.CALLS)
    bridge_role = next(r for r in roles if r.node_id == "bridge")
    # Bridge has high betweenness and moderate degree
    assert bridge_role.role in (StructuralRole.BRIDGE, StructuralRole.HUB)


def test_utility_detection():
    """A node called by many but calling few should be a utility."""
    g = CodeGraph()
    g.add_node(Node(id="util", kind=NodeKind.FUNCTION, file="m.py", line=1, name="util"))
    for i in range(5):
        nid = f"caller{i}"
        g.add_node(Node(id=nid, kind=NodeKind.FUNCTION, file="m.py", line=i+2, name=nid))
        g.add_edge(Edge(source=nid, target="util", kind=EdgeKind.CALLS))

    roles = classify_roles(g, edge_kind=EdgeKind.CALLS)
    util_role = next(r for r in roles if r.node_id == "util")
    assert util_role.role == StructuralRole.UTILITY
    assert util_role.in_degree == 5
    assert util_role.out_degree == 0


def test_entry_point_detection():
    """A node that calls many but is called by few should be an entry point."""
    g = CodeGraph()
    g.add_node(Node(id="main", kind=NodeKind.FUNCTION, file="m.py", line=1, name="main"))
    for i in range(5):
        nid = f"callee{i}"
        g.add_node(Node(id=nid, kind=NodeKind.FUNCTION, file="m.py", line=i+2, name=nid))
        g.add_edge(Edge(source="main", target=nid, kind=EdgeKind.CALLS))

    roles = classify_roles(g, edge_kind=EdgeKind.CALLS)
    main_role = next(r for r in roles if r.node_id == "main")
    assert main_role.role == StructuralRole.ENTRY_POINT
    assert main_role.out_degree == 5
    assert main_role.in_degree == 0


def test_regular_node():
    """A node with moderate connections should be regular."""
    g = CodeGraph()
    for name in ["a", "b", "c"]:
        g.add_node(Node(id=name, kind=NodeKind.FUNCTION, file="m.py", line=1, name=name))
    g.add_edge(Edge(source="a", target="b", kind=EdgeKind.CALLS))
    g.add_edge(Edge(source="b", target="c", kind=EdgeKind.CALLS))

    roles = classify_roles(g, edge_kind=EdgeKind.CALLS)
    b_role = next(r for r in roles if r.node_id == "b")
    assert b_role.role == StructuralRole.REGULAR
