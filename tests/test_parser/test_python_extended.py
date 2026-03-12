"""Extended tests for the Python parser — src layout and call resolution."""

import textwrap
from pathlib import Path

from topo_parser.graph import EdgeKind, NodeKind
from topo_parser.python import parse_python_project


def test_src_layout_module_ids(tmp_path: Path):
    """src-layout packages should produce correct Python module IDs."""
    src = tmp_path / "src" / "mypkg"
    src.mkdir(parents=True)
    (src / "__init__.py").write_text("")
    (src / "core.py").write_text(textwrap.dedent("""\
        def helper():
            pass
    """))

    graph = parse_python_project(tmp_path)

    assert "mypkg" in graph.nodes
    assert "mypkg.core" in graph.nodes
    assert "mypkg.core.helper" in graph.nodes


def test_call_resolution_within_module(tmp_path: Path):
    """Calls to functions in the same module should be resolved."""
    (tmp_path / "mod.py").write_text(textwrap.dedent("""\
        def helper():
            pass

        def main():
            helper()
    """))

    graph = parse_python_project(tmp_path)
    calls = graph.edges_by_kind(EdgeKind.CALLS)
    assert any(
        e.source == "mod.main" and e.target == "mod.helper"
        for e in calls
    )


def test_unresolvable_calls_are_dropped(tmp_path: Path):
    """Calls to external/unknown functions should not appear as edges."""
    (tmp_path / "mod.py").write_text(textwrap.dedent("""\
        def main():
            print("hello")
            some_unknown_function()
    """))

    graph = parse_python_project(tmp_path)
    calls = graph.edges_by_kind(EdgeKind.CALLS)
    # "print" and "some_unknown_function" are not graph nodes, so no call edges
    assert len(calls) == 0


def test_nested_package_structure(tmp_path: Path):
    """Nested packages produce correct hierarchical module IDs."""
    pkg = tmp_path / "pkg"
    sub = pkg / "sub"
    sub.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (sub / "__init__.py").write_text("")
    (sub / "deep.py").write_text("def f(): pass\n")

    graph = parse_python_project(tmp_path)
    assert "pkg.sub.deep" in graph.nodes
    assert "pkg.sub.deep.f" in graph.nodes


def test_recursive_call_resolution(tmp_path: Path):
    """A function calling itself should produce a self-loop edge."""
    (tmp_path / "mod.py").write_text(textwrap.dedent("""\
        def recurse():
            recurse()
    """))

    graph = parse_python_project(tmp_path)
    calls = graph.edges_by_kind(EdgeKind.CALLS)
    assert any(
        e.source == "mod.recurse" and e.target == "mod.recurse"
        for e in calls
    )


def test_cross_module_call_via_import(tmp_path: Path):
    """Calls to imported functions from another module should resolve."""
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "utils.py").write_text(textwrap.dedent("""\
        def do_stuff():
            pass
    """))
    (pkg / "main.py").write_text(textwrap.dedent("""\
        from pkg.utils import do_stuff

        def run():
            do_stuff()
    """))

    graph = parse_python_project(tmp_path)
    calls = graph.edges_by_kind(EdgeKind.CALLS)
    assert any(
        e.source == "pkg.main.run" and e.target == "pkg.utils.do_stuff"
        for e in calls
    ), f"Expected cross-module call, got: {[(e.source, e.target) for e in calls]}"


def test_from_import_creates_qualified_import_edges(tmp_path: Path):
    """'from foo import bar' should create import edges to both foo and foo.bar."""
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "lib.py").write_text("class Thing: pass\n")
    (pkg / "app.py").write_text("from pkg.lib import Thing\n")

    graph = parse_python_project(tmp_path)
    imports = graph.edges_by_kind(EdgeKind.IMPORTS)
    targets = [e.target for e in imports if e.source == "pkg.app"]
    assert "pkg.lib" in targets
    assert "pkg.lib.Thing" in targets
