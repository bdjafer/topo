"""CodeGraph JSON serialization and deserialization.

Thin wrappers around CodeGraph.to_dict() / from_dict() plus file I/O helpers.
"""

from __future__ import annotations

import json
from pathlib import Path

from topo_parser.graph import CodeGraph


def serialize_graph(graph: CodeGraph) -> dict:
    """Convert a CodeGraph to a JSON-serializable dict."""
    return graph.to_dict()


def deserialize_graph(data: dict) -> CodeGraph:
    """Reconstruct a CodeGraph from a JSON dict."""
    return CodeGraph.from_dict(data)


def save_graph(graph: CodeGraph, path: Path) -> None:
    """Serialize a CodeGraph and write it to a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(graph.to_dict(), indent=2) + "\n")


def load_graph(path: Path) -> CodeGraph:
    """Load a CodeGraph from a JSON file."""
    return CodeGraph.from_dict(json.loads(path.read_text()))
