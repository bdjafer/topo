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


def test_self_method_call_resolved(tmp_path: Path):
    """self.method() calls should resolve to the method on the same class."""
    (tmp_path / "mod.py").write_text(textwrap.dedent("""\
        class MyClass:
            def helper(self):
                pass

            def run(self):
                self.helper()
    """))

    graph = parse_python_project(tmp_path)
    calls = graph.edges_by_kind(EdgeKind.CALLS)
    assert any(
        e.source == "mod.MyClass.run" and e.target == "mod.MyClass.helper"
        for e in calls
    ), f"Expected self.helper() to resolve, got: {[(e.source, e.target) for e in calls]}"


def test_cls_method_call_resolved(tmp_path: Path):
    """cls.method() calls in classmethods should resolve."""
    (tmp_path / "mod.py").write_text(textwrap.dedent("""\
        class MyClass:
            @classmethod
            def create(cls):
                return cls._build()

            @classmethod
            def _build(cls):
                pass
    """))

    graph = parse_python_project(tmp_path)
    calls = graph.edges_by_kind(EdgeKind.CALLS)
    assert any(
        e.source == "mod.MyClass.create" and e.target == "mod.MyClass._build"
        for e in calls
    ), f"Expected cls._build() to resolve, got: {[(e.source, e.target) for e in calls]}"


def test_super_method_call_resolves_to_base(tmp_path: Path):
    """super().method() should resolve to the BASE class's method, not create a self-loop."""
    (tmp_path / "mod.py").write_text(textwrap.dedent("""\
        class Base:
            def setup(self):
                pass

        class Child(Base):
            def setup(self):
                super().setup()
    """))

    graph = parse_python_project(tmp_path)
    calls = graph.edges_by_kind(EdgeKind.CALLS)
    # Must resolve to Base.setup, NOT Child.setup (which would be a self-loop)
    assert any(
        e.source == "mod.Child.setup" and e.target == "mod.Base.setup"
        for e in calls
    ), f"Expected super().setup() -> Base.setup, got: {[(e.source, e.target) for e in calls]}"
    # Must NOT create a self-loop
    assert not any(
        e.source == "mod.Child.setup" and e.target == "mod.Child.setup"
        for e in calls
    ), "super().setup() should not create a self-loop"


def test_super_with_no_base_class_dropped(tmp_path: Path):
    """super() in a class with no explicit base should be dropped."""
    (tmp_path / "mod.py").write_text(textwrap.dedent("""\
        class MyClass:
            def run(self):
                super().__init__()
    """))

    graph = parse_python_project(tmp_path)
    calls = graph.edges_by_kind(EdgeKind.CALLS)
    assert len(calls) == 0, (
        f"super() with no base class should produce no calls, got: "
        f"{[(e.source, e.target) for e in calls]}"
    )


def test_self_call_to_nonexistent_method_dropped(tmp_path: Path):
    """self.method() where the method doesn't exist should be dropped."""
    (tmp_path / "mod.py").write_text(textwrap.dedent("""\
        class MyClass:
            def run(self):
                self.nonexistent_method()
    """))

    graph = parse_python_project(tmp_path)
    calls = graph.edges_by_kind(EdgeKind.CALLS)
    # nonexistent_method is not a node, so the edge should be dropped
    assert len(calls) == 0


def test_self_calls_across_multiple_classes(tmp_path: Path):
    """self.method() should resolve to the correct class when multiple exist."""
    (tmp_path / "mod.py").write_text(textwrap.dedent("""\
        class Alpha:
            def work(self):
                pass

            def run(self):
                self.work()

        class Beta:
            def work(self):
                pass

            def run(self):
                self.work()
    """))

    graph = parse_python_project(tmp_path)
    calls = graph.edges_by_kind(EdgeKind.CALLS)
    # Alpha.run -> Alpha.work (not Beta.work)
    assert any(
        e.source == "mod.Alpha.run" and e.target == "mod.Alpha.work"
        for e in calls
    )
    # Beta.run -> Beta.work (not Alpha.work)
    assert any(
        e.source == "mod.Beta.run" and e.target == "mod.Beta.work"
        for e in calls
    )
    # No cross-class calls
    assert not any(
        e.source == "mod.Alpha.run" and e.target == "mod.Beta.work"
        for e in calls
    )


def test_self_call_does_not_affect_free_functions(tmp_path: Path):
    """Free functions with 'self' as a regular parameter should not get special treatment."""
    (tmp_path / "mod.py").write_text(textwrap.dedent("""\
        def helper():
            pass

        def not_a_method():
            helper()
    """))

    graph = parse_python_project(tmp_path)
    calls = graph.edges_by_kind(EdgeKind.CALLS)
    # Free function calls should still work as before
    assert any(
        e.source == "mod.not_a_method" and e.target == "mod.helper"
        for e in calls
    )


def test_duplicate_self_calls_deduplicated(tmp_path: Path):
    """Calling self.method() multiple times should produce only one edge."""
    (tmp_path / "mod.py").write_text(textwrap.dedent("""\
        class MyClass:
            def helper(self):
                pass

            def run(self):
                self.helper()
                self.helper()
                self.helper()
    """))

    graph = parse_python_project(tmp_path)
    calls = graph.edges_by_kind(EdgeKind.CALLS)
    matching = [e for e in calls if e.source == "mod.MyClass.run" and e.target == "mod.MyClass.helper"]
    assert len(matching) == 1, f"Expected 1 edge, got {len(matching)}"


def test_inherited_method_call_is_dropped(tmp_path: Path):
    """self.method() where method is only on a parent class is not resolved.

    This is a known limitation: the parser resolves self.method() by looking
    up ClassName.method in the graph. If the method is inherited (exists on
    Base but not on Child), it won't be found. Fixing this would require
    walking the inheritance chain during resolution.
    """
    (tmp_path / "mod.py").write_text(textwrap.dedent("""\
        class Base:
            def base_method(self):
                pass

        class Child(Base):
            def run(self):
                self.base_method()
    """))

    graph = parse_python_project(tmp_path)
    calls = graph.edges_by_kind(EdgeKind.CALLS)
    # base_method is on Base, not Child. self.base_method() in Child emits
    # "Child.base_method" which won't resolve. This documents the limitation.
    assert len(calls) == 0, (
        f"Inherited method calls shouldn't resolve yet (known limitation), "
        f"got: {[(e.source, e.target) for e in calls]}"
    )


def test_staticmethod_not_affected(tmp_path: Path):
    """Static methods (no self/cls param) should not get self-resolution."""
    (tmp_path / "mod.py").write_text(textwrap.dedent("""\
        class MyClass:
            def helper(self):
                pass

            @staticmethod
            def run():
                pass
    """))

    graph = parse_python_project(tmp_path)
    calls = graph.edges_by_kind(EdgeKind.CALLS)
    assert len(calls) == 0
