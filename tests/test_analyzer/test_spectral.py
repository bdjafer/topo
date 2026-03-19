"""Tests for spectral decomposition."""

from topo_parser.graph import CodeGraph, Edge, EdgeKind, Node, NodeKind
from topo_analyzer.spectral import spectral_decomposition


def _make_chain_graph(n: int) -> CodeGraph:
    """Create a simple chain graph: f0 → f1 → f2 → ... → f(n-1)."""
    g = CodeGraph()
    for i in range(n):
        g.add_node(Node(id=f"f{i}", kind=NodeKind.FUNCTION, file="m.py", line=i, name=f"f{i}"))
    for i in range(n - 1):
        g.add_edge(Edge(source=f"f{i}", target=f"f{i+1}", kind=EdgeKind.CALLS))
    return g


def test_spectral_returns_result_for_valid_graph():
    g = _make_chain_graph(10)
    result = spectral_decomposition(g, edge_kind=EdgeKind.CALLS, k=4)
    assert result is not None
    assert len(result.eigenvalues) == 4
    assert result.eigenvectors.shape == (10, 4)
    assert result.fiedler_value > 0


def test_spectral_returns_none_for_tiny_graph():
    g = _make_chain_graph(2)
    result = spectral_decomposition(g, edge_kind=EdgeKind.CALLS, k=4)
    assert result is None


def test_fingerprints_differ():
    """Nodes in different positions should have different fingerprints."""
    g = _make_chain_graph(10)
    result = spectral_decomposition(g, edge_kind=EdgeKind.CALLS, k=4)
    assert result is not None
    fp0 = result.fingerprint("f0")
    fp5 = result.fingerprint("f5")
    # Endpoints and midpoints should have distinct fingerprints
    assert not (fp0 == fp5).all()
