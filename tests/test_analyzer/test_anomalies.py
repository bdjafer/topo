"""Tests for structural anomaly detection."""

from pathlib import Path

import numpy as np

from topo_parser.graph import CodeGraph, Edge, EdgeKind, Node, NodeKind
from topo_analyzer.anomalies import (
    AnomalyKind,
    detect_anomalies,
    _detect_cross_module,
    _detect_spectral_outliers,
    _detect_cycles,
    _detect_layer_discrepancies,
)
from topo_analyzer.modules import Module
from topo_analyzer.spectral import SpectralComponent, SpectralResult


def test_cross_module_anomaly():
    """A bidirectional cross-module dependency should be flagged."""
    g = CodeGraph()
    g.add_node(Node(id="a", kind=NodeKind.FUNCTION, file=Path("a.py"), line=1, name="a"))
    g.add_node(Node(id="b", kind=NodeKind.FUNCTION, file=Path("b.py"), line=1, name="b"))
    g.add_edge(Edge(source="a", target="b", kind=EdgeKind.CALLS))
    g.add_edge(Edge(source="b", target="a", kind=EdgeKind.CALLS))

    modules = [
        Module(id=0, node_ids=["a"]),
        Module(id=1, node_ids=["b"]),
    ]

    anomalies = _detect_cross_module(g, modules, EdgeKind.CALLS)
    assert len(anomalies) == 1
    assert anomalies[0].kind == AnomalyKind.CROSS_MODULE
    assert anomalies[0].edge_counts[EdgeKind.CALLS] == 2


def test_no_cross_module_within_same_module():
    """An edge within the same module should not be flagged."""
    g = CodeGraph()
    g.add_node(Node(id="a", kind=NodeKind.FUNCTION, file=Path("a.py"), line=1, name="a"))
    g.add_node(Node(id="b", kind=NodeKind.FUNCTION, file=Path("b.py"), line=1, name="b"))
    g.add_edge(Edge(source="a", target="b", kind=EdgeKind.CALLS))

    modules = [Module(id=0, node_ids=["a", "b"])]
    anomalies = _detect_cross_module(g, modules, EdgeKind.CALLS)
    assert len(anomalies) == 0


def test_cross_module_severity_penalizes_reverse_direction():
    """More balanced reverse flow should score as more severe."""
    g = CodeGraph()
    for name in ["a0", "a1", "a2", "b0", "b1", "c0", "c1", "d0", "d1"]:
        g.add_node(Node(id=name, kind=NodeKind.FUNCTION, file=Path("m.py"), line=1, name=name))

    for source, target in [
        ("a0", "b0"), ("a1", "b0"), ("a2", "b1"), ("b1", "a0"),
        ("c0", "d0"), ("d0", "c0"), ("c1", "d1"), ("d1", "c1"),
    ]:
        g.add_edge(Edge(source=source, target=target, kind=EdgeKind.CALLS))

    modules = [
        Module(id=0, node_ids=["a0", "a1", "a2"]),
        Module(id=1, node_ids=["b0", "b1"]),
        Module(id=2, node_ids=["c0", "c1"]),
        Module(id=3, node_ids=["d0", "d1"]),
    ]

    anomalies = _detect_cross_module(g, modules, EdgeKind.CALLS)
    severity_by_pair = {a.description.split(":")[0]: a.severity for a in anomalies}

    assert severity_by_pair["Bidirectional dependency between module 2 and module 3"] > (
        severity_by_pair["Bidirectional dependency between module 0 and module 1"]
    )


def test_cycle_detection():
    """A strongly connected dependency group should be detected."""
    g = CodeGraph()
    for name in ["a", "b", "c"]:
        g.add_node(Node(id=name, kind=NodeKind.FUNCTION, file=Path("m.py"), line=1, name=name))
    g.add_edge(Edge(source="a", target="b", kind=EdgeKind.CALLS))
    g.add_edge(Edge(source="b", target="c", kind=EdgeKind.CALLS))
    g.add_edge(Edge(source="c", target="a", kind=EdgeKind.CALLS))

    anomalies = _detect_cycles(g, EdgeKind.CALLS)
    assert len(anomalies) == 1
    cycle_anomaly = anomalies[0]
    assert cycle_anomaly.kind == AnomalyKind.CYCLE_MEMBER
    assert set(cycle_anomaly.node_ids) == {"a", "b", "c"}


def test_no_cycles_in_dag():
    """A DAG should produce no cycle anomalies."""
    g = CodeGraph()
    for name in ["a", "b", "c"]:
        g.add_node(Node(id=name, kind=NodeKind.FUNCTION, file=Path("m.py"), line=1, name=name))
    g.add_edge(Edge(source="a", target="b", kind=EdgeKind.CALLS))
    g.add_edge(Edge(source="b", target="c", kind=EdgeKind.CALLS))

    anomalies = _detect_cycles(g, EdgeKind.CALLS)
    assert len(anomalies) == 0


def test_detect_anomalies_integration():
    """Full anomaly detection pipeline runs without errors."""
    g = CodeGraph()
    g.add_node(Node(id="a", kind=NodeKind.FUNCTION, file=Path("m.py"), line=1, name="a"))
    g.add_node(Node(id="b", kind=NodeKind.FUNCTION, file=Path("m.py"), line=2, name="b"))
    g.add_edge(Edge(source="a", target="b", kind=EdgeKind.CALLS))

    anomalies = detect_anomalies(g, spectral=None, modules=[], edge_kind=EdgeKind.CALLS)
    assert isinstance(anomalies, list)


def test_spectral_outlier_detected_within_module():
    """Spectral outliers should be scored relative to their own module."""
    spectral = SpectralResult(
        total_node_ids=["a", "b", "c", "d"],
        components=[
            SpectralComponent(
                id=0,
                node_ids=["a", "b", "c", "d"],
                eigenvalues=np.array([0.1, 0.2]),
                eigenvectors=np.array([
                    [0.0, 0.0],
                    [0.1, 0.0],
                    [0.0, 0.1],
                    [3.0, 3.0],
                ]),
            )
        ],
        unassigned_components=[],
        primary_eigenvalues=np.array([0.1, 0.2]),
        component_sizes=[4],
        fiedler_value=0.1,
    )
    modules = [Module(id=0, node_ids=["a", "b", "c", "d"], confidence=0.8)]

    anomalies = _detect_spectral_outliers(spectral, modules, projection=None, threshold_sigma=1.5)
    assert len(anomalies) == 1
    assert anomalies[0].kind == AnomalyKind.SPECTRAL_OUTLIER
    assert anomalies[0].node_ids == ["d"]


def test_spectral_outlier_multi_component_no_crash():
    """Outlier detection should handle components with different fingerprint widths."""
    spectral = SpectralResult(
        total_node_ids=["a0", "a1", "a2", "a3", "b0", "b1", "b2"],
        components=[
            SpectralComponent(
                id=0,
                node_ids=["a0", "a1", "a2", "a3"],
                eigenvalues=np.array([0.1, 0.2, 0.3]),
                eigenvectors=np.array([
                    [0.0, 0.0, 0.0],
                    [0.1, 0.0, 0.0],
                    [0.0, 0.1, 0.0],
                    [3.0, 3.0, 3.0],
                ]),
            ),
            SpectralComponent(
                id=1,
                node_ids=["b0", "b1", "b2"],
                eigenvalues=np.array([0.5]),
                eigenvectors=np.array([
                    [0.0],
                    [0.1],
                    [5.0],
                ]),
            ),
        ],
        unassigned_components=[],
        primary_eigenvalues=np.array([0.1, 0.2, 0.3]),
        component_sizes=[4, 3],
        fiedler_value=0.1,
    )
    modules = [
        Module(id=0, node_ids=["a0", "a1", "a2", "a3"], confidence=0.8, component_id=0),
        Module(id=1, node_ids=["b0", "b1", "b2"], confidence=0.7, component_id=1),
    ]

    # Should not crash despite different eigenvector widths (3 vs 1)
    anomalies = _detect_spectral_outliers(spectral, modules, projection=None, threshold_sigma=1.5)
    assert isinstance(anomalies, list)
    outlier_nodes = {nid for a in anomalies for nid in a.node_ids}
    assert "a3" in outlier_nodes or "b2" in outlier_nodes


def test_layer_discrepancy_per_kind_normalization():
    """MODULE nodes with high imports and zero calls should not be flagged.

    The detector computes percentiles within each NodeKind group, so
    MODULE nodes are compared to other MODULEs — making the typical
    Python pattern (high imports, zero calls) unremarkable.
    """
    g = CodeGraph()
    # 4 MODULE nodes: all have high imports, zero calls (normal Python pattern)
    for i in range(4):
        g.add_node(Node(id=f"mod{i}", kind=NodeKind.MODULE, file=Path(f"mod{i}.py"), line=1, name=f"mod{i}"))
    for i in range(4):
        for j in range(4):
            if i != j:
                g.add_edge(Edge(source=f"mod{i}", target=f"mod{j}", kind=EdgeKind.IMPORTS))

    # 4 FUNCTION nodes: all have calls, zero imports (normal Python pattern)
    for i in range(4):
        g.add_node(Node(id=f"fn{i}", kind=NodeKind.FUNCTION, file=Path(f"fn{i}.py"), line=1, name=f"fn{i}"))
    for i in range(4):
        for j in range(4):
            if i != j:
                g.add_edge(Edge(source=f"fn{i}", target=f"fn{j}", kind=EdgeKind.CALLS))

    anomalies = _detect_layer_discrepancies(g, [EdgeKind.CALLS, EdgeKind.IMPORTS])

    # No discrepancies: within their kind, each group has uniform degrees
    # across the layer they participate in, and the min_deg guard excludes
    # nodes absent from a layer.
    assert len(anomalies) == 0


def test_layer_discrepancy_flags_genuine_cross_layer_tension():
    """A FUNCTION node with high calls AND high imports but very different
    percentiles within its kind group should be flagged."""
    g = CodeGraph()
    # 5 functions: fn0-fn3 have moderate calls, fn4 has extreme calls
    for i in range(5):
        g.add_node(Node(id=f"fn{i}", kind=NodeKind.FUNCTION, file=Path(f"fn.py"), line=i, name=f"fn{i}"))
    # fn0-fn3: each has 1 call edge and 4 import edges
    for i in range(4):
        g.add_edge(Edge(source=f"fn{i}", target=f"fn{(i+1)%4}", kind=EdgeKind.CALLS))
        for j in range(4):
            if i != j:
                g.add_edge(Edge(source=f"fn{i}", target=f"fn{j}", kind=EdgeKind.IMPORTS))
    # fn4: has 4 call edges but only 2 import edges — calls-heavy, imports-light
    for i in range(4):
        g.add_edge(Edge(source=f"fn4", target=f"fn{i}", kind=EdgeKind.CALLS))
    g.add_edge(Edge(source=f"fn4", target=f"fn0", kind=EdgeKind.IMPORTS))
    g.add_edge(Edge(source=f"fn4", target=f"fn1", kind=EdgeKind.IMPORTS))

    anomalies = _detect_layer_discrepancies(g, [EdgeKind.CALLS, EdgeKind.IMPORTS])

    # fn4 should be flagged: calls-central (highest among FUNCTIONs) but
    # imports-peripheral (lowest among FUNCTIONs)
    flagged = {nid for a in anomalies for nid in a.node_ids}
    assert "fn4" in flagged


def test_layer_discrepancy_absent_layer_skipped():
    """A node with degree < 2 in one layer should not be flagged."""
    g = CodeGraph()
    for i in range(5):
        g.add_node(Node(id=f"fn{i}", kind=NodeKind.FUNCTION, file=Path(f"fn.py"), line=i, name=f"fn{i}"))
    # All functions have calls edges
    for i in range(5):
        g.add_edge(Edge(source=f"fn{i}", target=f"fn{(i+1)%5}", kind=EdgeKind.CALLS))
    # Only fn0 has import edges (degree 0 in imports for fn1-fn4)
    for i in range(1, 5):
        g.add_edge(Edge(source="fn0", target=f"fn{i}", kind=EdgeKind.IMPORTS))

    anomalies = _detect_layer_discrepancies(g, [EdgeKind.CALLS, EdgeKind.IMPORTS])

    # fn1-fn4 have imports_degree=1 (inbound from fn0) and calls_degree=2
    # fn0 has imports_degree=4 and calls_degree=2
    # fn1-fn4 are filtered by min_deg < 2 guard (imports_degree=1).
    # fn0 passes min_deg but has no large percentile gap.
    for a in anomalies:
        for nid in a.node_ids:
            # Every flagged node should have degree >= 2 in both layers
            node_calls_deg = sum(1 for e in g.edges_by_kind(EdgeKind.CALLS)
                                 if e.source == nid or e.target == nid)
            node_imports_deg = sum(1 for e in g.edges_by_kind(EdgeKind.IMPORTS)
                                  if e.source == nid or e.target == nid)
            assert node_calls_deg >= 2
            assert node_imports_deg >= 2


def test_layer_discrepancy_small_kind_group_skipped():
    """A kind group with fewer than 5 nodes produces no findings."""
    g = CodeGraph()
    # Only 2 MODULE nodes (too few for percentiles)
    g.add_node(Node(id="m0", kind=NodeKind.MODULE, file=Path("m0.py"), line=1, name="m0"))
    g.add_node(Node(id="m1", kind=NodeKind.MODULE, file=Path("m1.py"), line=1, name="m1"))
    g.add_edge(Edge(source="m0", target="m1", kind=EdgeKind.IMPORTS))
    g.add_edge(Edge(source="m0", target="m1", kind=EdgeKind.CALLS))
    g.add_edge(Edge(source="m1", target="m0", kind=EdgeKind.CALLS))

    anomalies = _detect_layer_discrepancies(g, [EdgeKind.CALLS, EdgeKind.IMPORTS])
    assert len(anomalies) == 0
