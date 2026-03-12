"""Tests for module detection via spectral clustering."""

from pathlib import Path

from topo_parser.graph import CodeGraph, Edge, EdgeKind, Node, NodeKind
from topo_analyzer.spectral import spectral_decomposition
from topo_analyzer.modules import detect_modules, _estimate_k


def _make_two_cluster_graph() -> CodeGraph:
    """Create a graph with two clear clusters connected by a single bridge edge."""
    g = CodeGraph()
    # Cluster A: tightly connected
    for i in range(5):
        g.add_node(Node(id=f"a{i}", kind=NodeKind.FUNCTION, file=Path("a.py"), line=i, name=f"a{i}"))
    for i in range(5):
        for j in range(i + 1, 5):
            g.add_edge(Edge(source=f"a{i}", target=f"a{j}", kind=EdgeKind.CALLS))

    # Cluster B: tightly connected
    for i in range(5):
        g.add_node(Node(id=f"b{i}", kind=NodeKind.FUNCTION, file=Path("b.py"), line=i, name=f"b{i}"))
    for i in range(5):
        for j in range(i + 1, 5):
            g.add_edge(Edge(source=f"b{i}", target=f"b{j}", kind=EdgeKind.CALLS))

    # Single bridge edge
    g.add_edge(Edge(source="a0", target="b0", kind=EdgeKind.CALLS))
    return g


def test_detect_two_modules():
    """Two well-separated clusters should be detected as 2 modules."""
    g = _make_two_cluster_graph()
    spectral = spectral_decomposition(g, edge_kind=EdgeKind.CALLS, k=4)
    assert spectral is not None

    modules = detect_modules(spectral, n_modules=2)
    assert len(modules) == 2

    # Both modules should have members (non-trivial split)
    sizes = sorted(m.size for m in modules)
    assert all(s > 0 for s in sizes)
    assert sum(sizes) == 10


def test_detect_modules_auto_k():
    """Module detection with automatic k estimation."""
    g = _make_two_cluster_graph()
    spectral = spectral_decomposition(g, edge_kind=EdgeKind.CALLS, k=4)
    assert spectral is not None

    modules = detect_modules(spectral)  # auto k
    assert len(modules) >= 2


def test_estimate_k_single_eigenvalue():
    """With a single eigenvalue, should return 2."""
    import numpy as np
    assert _estimate_k(np.array([0.5])) == 2
