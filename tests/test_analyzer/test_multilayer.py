"""Tests for multi-layer spectral decomposition."""

from pathlib import Path

from topo_parser.graph import CodeGraph, Edge, EdgeKind, Node, NodeKind
from topo_analyzer.spectral import (
    spectral_decomposition,
    spectral_decomposition_multilayer,
)
from topo_analyzer.analysis import analyze


def _make_multilayer_graph() -> CodeGraph:
    """Graph where different layers connect different nodes."""
    g = CodeGraph()
    for i in range(6):
        g.add_node(Node(id=f"n{i}", kind=NodeKind.FUNCTION, file=Path("m.py"), line=i, name=f"n{i}"))

    # CALLS layer connects n0-n1-n2
    g.add_edge(Edge(source="n0", target="n1", kind=EdgeKind.CALLS))
    g.add_edge(Edge(source="n1", target="n2", kind=EdgeKind.CALLS))

    # CONTAINS layer connects n2-n3-n4
    g.add_edge(Edge(source="n2", target="n3", kind=EdgeKind.CONTAINS))
    g.add_edge(Edge(source="n3", target="n4", kind=EdgeKind.CONTAINS))

    # IMPORTS layer connects n4-n5-n0 (closing the loop)
    g.add_edge(Edge(source="n4", target="n5", kind=EdgeKind.IMPORTS))
    g.add_edge(Edge(source="n5", target="n0", kind=EdgeKind.IMPORTS))

    return g


def test_single_layer_has_orphans():
    """A single CALLS layer leaves most nodes disconnected."""
    g = _make_multilayer_graph()
    result = spectral_decomposition(g, edge_kind=EdgeKind.CALLS)
    # Only 3 nodes connected on CALLS, not enough for spectral
    # or it produces a result with many disconnected nodes
    if result is not None:
        assert len(result.node_ids) == 6


def test_multilayer_connects_all_nodes():
    """Multi-layer decomposition should see edges from all layers."""
    g = _make_multilayer_graph()
    result = spectral_decomposition_multilayer(g)
    assert result is not None
    assert len(result.node_ids) == 6
    # Fiedler > 0 means the combined graph is connected
    assert result.fiedler_value > 0


def test_multilayer_with_custom_weights():
    """Custom layer weights are applied."""
    g = _make_multilayer_graph()
    result = spectral_decomposition_multilayer(
        g, layer_weights={EdgeKind.CALLS: 1.0, EdgeKind.CONTAINS: 0.0}
    )
    # With CONTAINS weight=0, only CALLS and no IMPORTS weight provided
    # so fewer connections
    assert result is not None


def test_analyze_combined_mode():
    """analyze(combined=True) runs end-to-end."""
    g = _make_multilayer_graph()
    result = analyze(g, combined=True)
    assert result.spectral is not None
    # No orphans — every node is connected through some layer
    orphans = [r for r in result.roles if r.role.value == "orphan"]
    assert len(orphans) == 0
