"""Tests for CodeGraph JSON serialization round-trip."""

from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile

from topo_parser.graph import CodeGraph, Edge, EdgeKind, Node, NodeKind
from topo_benchmark.codegraph_io import deserialize_graph, load_graph, save_graph, serialize_graph


def _make_graph() -> CodeGraph:
    g = CodeGraph()
    g.add_node(Node(id="pkg.mod.fn", kind=NodeKind.FUNCTION, file="pkg/mod.py", line=12, name="fn"))
    g.add_node(Node(id="pkg.mod.cls", kind=NodeKind.CLASS, file="pkg/mod.py", line=1, name="cls"))
    g.add_node(Node(id="pkg.mod", kind=NodeKind.MODULE, file="pkg/mod.py", line=0, name="mod"))
    g.add_edge(Edge(source="pkg.mod.fn", target="pkg.mod.cls", kind=EdgeKind.CALLS))
    g.add_edge(Edge(source="pkg.mod", target="pkg.mod.fn", kind=EdgeKind.CONTAINS))
    g.add_edge(Edge(source="pkg.mod.cls", target="pkg.mod.fn", kind=EdgeKind.INHERITS))
    return g


def test_round_trip_serialize_deserialize():
    original = _make_graph()
    data = serialize_graph(original)
    restored = deserialize_graph(data)

    assert set(restored.nodes.keys()) == set(original.nodes.keys())
    for nid in original.nodes:
        assert restored.nodes[nid].kind == original.nodes[nid].kind
        assert restored.nodes[nid].name == original.nodes[nid].name
        assert restored.nodes[nid].line == original.nodes[nid].line

    assert len(restored.edges) == len(original.edges)
    orig_edges = {(e.source, e.target, e.kind) for e in original.edges}
    rest_edges = {(e.source, e.target, e.kind) for e in restored.edges}
    assert orig_edges == rest_edges


def test_round_trip_file_io(tmp_path: Path):
    original = _make_graph()
    path = tmp_path / "graph.json"
    save_graph(original, path)
    restored = load_graph(path)

    assert set(restored.nodes.keys()) == set(original.nodes.keys())
    assert len(restored.edges) == len(original.edges)


def test_serialized_format():
    g = _make_graph()
    data = serialize_graph(g)

    assert "nodes" in data
    assert "edges" in data
    assert all("id" in n and "kind" in n and "file" in n for n in data["nodes"])
    assert all("source" in e and "target" in e and "kind" in e for e in data["edges"])
    # kind should be string values, not enum objects
    assert all(isinstance(n["kind"], str) for n in data["nodes"])
    assert all(isinstance(e["kind"], str) for e in data["edges"])
