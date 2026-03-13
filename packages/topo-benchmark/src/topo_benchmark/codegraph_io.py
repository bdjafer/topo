"""CodeGraph JSON serialization and deserialization."""

from __future__ import annotations

import json
from pathlib import Path

from topo_parser.graph import CodeGraph, Edge, EdgeKind, Node, NodeKind


def serialize_graph(graph: CodeGraph) -> dict:
    """Convert a CodeGraph to a JSON-serializable dict."""
    nodes = []
    for node in sorted(graph.nodes.values(), key=lambda n: n.id):
        nodes.append({
            "id": node.id,
            "kind": node.kind.value,
            "file": str(node.file),
            "line": node.line,
            "name": node.name,
        })
    edges = []
    for edge in graph.edges:
        edges.append({
            "source": edge.source,
            "target": edge.target,
            "kind": edge.kind.value,
        })
    return {"nodes": nodes, "edges": edges}


def deserialize_graph(data: dict) -> CodeGraph:
    """Reconstruct a CodeGraph from a JSON dict."""
    graph = CodeGraph()
    for nd in data["nodes"]:
        graph.add_node(Node(
            id=nd["id"],
            kind=NodeKind(nd["kind"]),
            file=Path(nd["file"]),
            line=nd["line"],
            name=nd["name"],
        ))
    for ed in data["edges"]:
        graph.add_edge(Edge(
            source=ed["source"],
            target=ed["target"],
            kind=EdgeKind(ed["kind"]),
        ))
    return graph


def save_graph(graph: CodeGraph, path: Path) -> None:
    """Serialize a CodeGraph and write it to a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(serialize_graph(graph), indent=2) + "\n")


def load_graph(path: Path) -> CodeGraph:
    """Load a CodeGraph from a JSON file."""
    return deserialize_graph(json.loads(path.read_text()))
