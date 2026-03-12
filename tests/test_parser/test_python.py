"""Tests for the Python parser."""

import textwrap
from pathlib import Path

from topo_parser.graph import EdgeKind, NodeKind
from topo_parser.python import parse_python_project


def test_parse_simple_module(tmp_path: Path):
    """Parse a single Python file with a function and a class."""
    (tmp_path / "example.py").write_text(textwrap.dedent("""\
        import os

        def helper():
            pass

        class MyClass:
            def method(self):
                helper()
    """))

    graph = parse_python_project(tmp_path)

    # Should have module, function, class, and method nodes
    assert graph.node_count >= 4

    # Check node kinds
    kinds = {n.kind for n in graph.nodes.values()}
    assert NodeKind.MODULE in kinds
    assert NodeKind.FUNCTION in kinds
    assert NodeKind.CLASS in kinds

    # Should have import edge for 'os'
    import_edges = graph.edges_by_kind(EdgeKind.IMPORTS)
    assert any(e.target == "os" for e in import_edges)

    # Should have call edge from method to helper (fully qualified)
    call_edges = graph.edges_by_kind(EdgeKind.CALLS)
    assert any(e.target == "example.helper" for e in call_edges)

    # Should have containment edges
    contain_edges = graph.edges_by_kind(EdgeKind.CONTAINS)
    assert len(contain_edges) >= 3  # module→class, module→helper, class→method


def test_parse_inheritance(tmp_path: Path):
    """Inheritance edges are captured."""
    (tmp_path / "animals.py").write_text(textwrap.dedent("""\
        class Animal:
            pass

        class Dog(Animal):
            pass
    """))

    graph = parse_python_project(tmp_path)
    inherit_edges = graph.edges_by_kind(EdgeKind.INHERITS)
    assert any(e.source == "animals.Dog" and e.target == "animals.Animal" for e in inherit_edges)


def test_parse_empty_dir(tmp_path: Path):
    """Parsing an empty directory produces an empty graph."""
    graph = parse_python_project(tmp_path)
    assert graph.node_count == 0
    assert graph.edge_count == 0
