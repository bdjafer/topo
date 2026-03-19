"""Tests for multi-layer spectral decomposition."""

from topo_parser.graph import CodeGraph, Edge, EdgeKind, Node, NodeKind
from topo_analyzer.spectral import (
    DEFAULT_LAYER_WEIGHTS,
    spectral_decomposition,
    spectral_decomposition_multilayer,
)
from topo_analyzer.analysis import analyze


def _make_multilayer_graph() -> CodeGraph:
    """Graph where different layers connect different nodes."""
    g = CodeGraph()
    for i in range(6):
        g.add_node(Node(id=f"n{i}", kind=NodeKind.FUNCTION, file="m.py", line=i, name=f"n{i}"))

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
    assert result is not None
    assert len(result.node_ids) == 3
    assert result.unassigned_node_ids == ["n3", "n4", "n5"]


def test_multilayer_connects_all_nodes():
    """Multi-layer decomposition should connect when all linking layers are weighted."""
    g = _make_multilayer_graph()
    result = spectral_decomposition_multilayer(
        g,
        layer_weights={
            EdgeKind.CALLS: 1.0,
            EdgeKind.IMPORTS: 1.0,
            EdgeKind.CONTAINS: 1.0,
        },
    )
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


def test_contains_weight_disabled_by_default():
    """Containment should not influence combined-mode clustering by default."""
    assert DEFAULT_LAYER_WEIGHTS[EdgeKind.CONTAINS] == 0.0


def test_analyze_combined_mode():
    """analyze(combined=True) runs end-to-end."""
    g = _make_multilayer_graph()
    result = analyze(g, combined=True)
    assert result.spectral is not None
    assert result.coverage is not None
    assert result.coverage.spectral_coverage_ratio < 1.0
    # CONTAINS is disabled by default, so n3 remains disconnected here.
    orphans = [r for r in result.roles if r.role.value == "orphan"]
    assert len(orphans) == 1
