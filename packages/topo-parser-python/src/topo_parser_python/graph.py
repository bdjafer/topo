"""
Core graph data model — the shared contract between all topo packages.

A codebase is represented as a typed, multilayer graph:
- Nodes are code entities (functions, classes, modules)
- Edges are structural relationships (calls, imports, inheritance, co-location)
- Each relationship type forms a separate layer
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class NodeKind(Enum):
    """Kind of code entity. Kept minimal — only distinctions that matter structurally."""

    MODULE = "module"
    CLASS = "class"
    FUNCTION = "function"


class EdgeKind(Enum):
    """Kind of structural relationship. Each kind forms a separate graph layer."""

    CALLS = "calls"  # A invokes B at runtime
    IMPORTS = "imports"  # A imports B
    INHERITS = "inherits"  # A extends/implements B
    CONTAINS = "contains"  # A structurally contains B (module→class, class→method)


@dataclass(frozen=True)
class Node:
    """A code entity in the graph."""

    id: str  # Fully qualified name, e.g. "pkg.module.ClassName.method"
    kind: NodeKind
    file: str  # Source file path (metadata, not a filesystem reference)
    line: int
    name: str  # Short name, e.g. "method"

    def __hash__(self) -> int:
        return hash(self.id)


@dataclass(frozen=True)
class Edge:
    """A directed structural relationship between two nodes."""

    source: str  # Node id
    target: str  # Node id
    kind: EdgeKind

    def __hash__(self) -> int:
        return hash((self.source, self.target, self.kind))


@dataclass
class CodeGraph:
    """
    A multilayer graph of a codebase.

    Nodes are code entities. Edges are structural relationships.
    Each EdgeKind forms a separate layer, all sharing the same node set.
    """

    nodes: dict[str, Node] = field(default_factory=dict)
    edges: list[Edge] = field(default_factory=list)

    def add_node(self, node: Node) -> None:
        self.nodes[node.id] = node

    def add_edge(self, edge: Edge) -> None:
        self.edges.append(edge)

    def edges_by_kind(self, kind: EdgeKind) -> list[Edge]:
        """Get all edges of a specific relationship type (one layer of the multilayer graph)."""
        return [e for e in self.edges if e.kind == kind]

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def edge_count(self) -> int:
        return len(self.edges)

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict — the canonical wire format."""
        return {
            "nodes": [
                {"id": n.id, "kind": n.kind.value, "file": n.file, "line": n.line, "name": n.name}
                for n in sorted(self.nodes.values(), key=lambda n: n.id)
            ],
            "edges": [
                {"source": e.source, "target": e.target, "kind": e.kind.value}
                for e in self.edges
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> CodeGraph:
        """Reconstruct from a dict produced by to_dict()."""
        graph = cls()
        for nd in data["nodes"]:
            graph.add_node(Node(
                id=nd["id"],
                kind=NodeKind(nd["kind"]),
                file=nd["file"],
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

    def summary(self) -> str:
        """Human-readable summary of graph contents."""
        lines = [f"CodeGraph: {self.node_count} nodes, {self.edge_count} edges"]
        for kind in NodeKind:
            count = sum(1 for n in self.nodes.values() if n.kind == kind)
            if count:
                lines.append(f"  {kind.value}: {count}")
        for kind in EdgeKind:
            count = sum(1 for e in self.edges if e.kind == kind)
            if count:
                lines.append(f"  {kind.value}: {count}")
        return "\n".join(lines)
