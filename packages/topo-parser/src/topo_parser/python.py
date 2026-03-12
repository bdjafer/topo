"""
Python source code parser.

Walks a Python codebase using the ast module and produces a CodeGraph
with nodes (modules, classes, functions) and edges (calls, imports,
inheritance, containment).
"""

from __future__ import annotations

import ast
from pathlib import Path

from topo_parser.graph import CodeGraph, Edge, EdgeKind, Node, NodeKind


def parse_python_project(root: Path) -> CodeGraph:
    """
    Parse a Python project directory into a CodeGraph.

    Walks all .py files under `root`, extracts entities and relationships.
    Detects src-layout packages and uses the Python package root for module IDs.
    """
    graph = CodeGraph()
    root = root.resolve()
    py_files = sorted(root.rglob("*.py"))

    # Detect Python package roots: directories containing __init__.py that
    # aren't nested inside another package. Handles both flat layout
    # (root/pkg/__init__.py) and src layout (root/src/pkg/__init__.py).
    package_roots = _find_package_roots(root, py_files)

    for py_file in py_files:
        try:
            source = py_file.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(py_file))
        except (SyntaxError, UnicodeDecodeError):
            continue

        pkg_root = _best_package_root(py_file, package_roots)
        _extract_from_file(graph, tree, py_file, pkg_root or root)

    _resolve_edges(graph)
    return graph


def _find_package_roots(root: Path, py_files: list[Path]) -> list[Path]:
    """Find directories that serve as Python package roots.

    A package root is the parent of a top-level Python package. For example,
    in a src layout (proj/src/pkg/__init__.py), the package root is proj/src/.
    In a flat layout (proj/pkg/__init__.py), the package root is proj/.
    """
    # Find all directories containing __init__.py
    init_dirs: set[Path] = set()
    for f in py_files:
        if f.name == "__init__.py":
            init_dirs.add(f.parent.resolve())

    # A package root is the parent of a package dir whose parent is NOT itself a package.
    # This gives us the directory from which Python module IDs should be computed.
    package_roots: set[Path] = set()
    for pkg_dir in init_dirs:
        parent = pkg_dir.parent
        if parent not in init_dirs:
            package_roots.add(parent)

    return sorted(package_roots)


def _best_package_root(file: Path, package_roots: list[Path]) -> Path | None:
    """Find the most specific package root that contains this file."""
    file = file.resolve()
    best: Path | None = None
    for pr in package_roots:
        if file.is_relative_to(pr):
            if best is None or len(pr.parts) > len(best.parts):
                best = pr
    return best


def _module_id(file: Path, root: Path) -> str:
    """Derive a module id from file path relative to root."""
    rel = file.relative_to(root)
    parts = list(rel.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts) if parts else rel.stem


def _extract_from_file(
    graph: CodeGraph, tree: ast.Module, file: Path, root: Path
) -> None:
    """Extract nodes and edges from a single parsed file."""
    mod_id = _module_id(file, root)

    # Module node
    graph.add_node(Node(id=mod_id, kind=NodeKind.MODULE, file=file, line=1, name=mod_id.split(".")[-1]))

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef):
            _extract_class(graph, node, mod_id, file)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            _extract_function(graph, node, mod_id, file)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            _extract_import(graph, node, mod_id)


def _extract_class(
    graph: CodeGraph, node: ast.ClassDef, parent_id: str, file: Path
) -> None:
    """Extract a class node, its methods, and inheritance edges."""
    class_id = f"{parent_id}.{node.name}"
    graph.add_node(Node(id=class_id, kind=NodeKind.CLASS, file=file, line=node.lineno, name=node.name))
    graph.add_edge(Edge(source=parent_id, target=class_id, kind=EdgeKind.CONTAINS))

    # Inheritance
    for base in node.bases:
        base_name = _resolve_name(base)
        if base_name:
            graph.add_edge(Edge(source=class_id, target=base_name, kind=EdgeKind.INHERITS))

    # Extract the first base class name for super() resolution
    first_base_name: str | None = None
    if node.bases:
        first_base_name = _resolve_name(node.bases[0])

    # Methods
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            _extract_function(
                graph, child, class_id, file,
                class_id=class_id, base_name=first_base_name,
            )


def _extract_function(
    graph: CodeGraph,
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    parent_id: str,
    file: Path,
    *,
    class_id: str | None = None,
    base_name: str | None = None,
) -> None:
    """Extract a function/method node and its call edges.

    Args:
        class_id: If this function is a method, the fully-qualified ID of
            the enclosing class.  Used to resolve self/cls method calls.
        base_name: The unresolved name of the first base class, used to
            resolve super().method() calls.
    """
    func_id = f"{parent_id}.{node.name}"
    graph.add_node(Node(id=func_id, kind=NodeKind.FUNCTION, file=file, line=node.lineno, name=node.name))
    graph.add_edge(Edge(source=parent_id, target=func_id, kind=EdgeKind.CONTAINS))

    # Determine the self/cls parameter name for methods
    self_name = _self_param_name(node) if class_id else None

    # Walk the function body for call expressions
    seen_calls: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            callee = _resolve_call(child.func, self_name, class_id, base_name)
            if callee and callee not in seen_calls:
                seen_calls.add(callee)
                graph.add_edge(Edge(source=func_id, target=callee, kind=EdgeKind.CALLS))


def _extract_import(graph: CodeGraph, node: ast.Import | ast.ImportFrom, mod_id: str) -> None:
    """Extract import edges.

    For `import foo.bar`, creates an edge to `foo.bar`.
    For `from foo.bar import baz, qux`, creates edges to both
    `foo.bar` (the module) and `foo.bar.baz`, `foo.bar.qux` (the names).
    This allows call resolution to map `baz()` -> `foo.bar.baz`.
    """
    if isinstance(node, ast.Import):
        for alias in node.names:
            graph.add_edge(Edge(source=mod_id, target=alias.name, kind=EdgeKind.IMPORTS))
    elif node.module:
        graph.add_edge(Edge(source=mod_id, target=node.module, kind=EdgeKind.IMPORTS))
        for alias in node.names:
            if alias.name != "*":
                qualified = f"{node.module}.{alias.name}"
                graph.add_edge(Edge(source=mod_id, target=qualified, kind=EdgeKind.IMPORTS))


def _self_param_name(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str | None:
    """Return the name of the first parameter if it looks like self/cls."""
    if node.args.args:
        name = node.args.args[0].arg
        if name in ("self", "cls"):
            return name
    return None


def _resolve_call(
    node: ast.expr,
    self_name: str | None,
    class_id: str | None,
    base_name: str | None = None,
) -> str | None:
    """Resolve a call expression, with special handling for self/cls/super().

    For ``self.method()``, rewrites to ``ClassName.method`` so the
    downstream resolver can match it to a known node via ancestor prefix walk.
    For ``super().method()``, rewrites to ``BaseName.method`` using the
    first declared base class.
    """
    # self.method() or cls.method() -> ClassName.method
    if (
        self_name
        and class_id
        and isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == self_name
    ):
        class_short = class_id.rsplit(".", 1)[-1]
        return f"{class_short}.{node.attr}"

    # super().method() -> BaseName.method
    if (
        class_id
        and isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "super"
    ):
        if base_name:
            # Use the declared base class name so the resolver can find
            # the parent's method rather than creating a self-loop.
            return f"{base_name}.{node.attr}"
        # No base class declared (implicit object) — drop the call
        return None

    return _resolve_name(node)


def _resolve_name(node: ast.expr) -> str | None:
    """Best-effort resolve an AST expression to a dotted name string."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        value = _resolve_name(node.value)
        if value:
            return f"{value}.{node.attr}"
    return None


def _resolve_edges(graph: CodeGraph) -> None:
    """Resolve raw targets in CALLS and INHERITS edges to actual graph node IDs.

    Raw edges have targets like 'helper' or 'Enum'. This pass replaces them
    with fully qualified node IDs where possible, and drops edges that can't
    be resolved to known nodes.
    """
    node_ids = set(graph.nodes)
    resolve_kinds = {EdgeKind.CALLS, EdgeKind.INHERITS}

    # Build import map: for each module, map imported short names to full targets.
    import_map: dict[str, dict[str, str]] = {}
    for edge in graph.edges_by_kind(EdgeKind.IMPORTS):
        mod = edge.source
        target = edge.target
        short = target.rsplit(".", 1)[-1]
        import_map.setdefault(mod, {})[short] = target

    keep: list[Edge] = []
    for edge in graph.edges:
        if edge.kind not in resolve_kinds:
            keep.append(edge)
            continue
        target = _try_resolve(edge.source, edge.target, node_ids, import_map)
        if target:
            keep.append(Edge(source=edge.source, target=target, kind=edge.kind))

    graph.edges = keep


def _try_resolve(
    source: str, raw_target: str, node_ids: set[str],
    import_map: dict[str, dict[str, str]],
) -> str | None:
    """Try to resolve a raw edge target to a known node ID."""
    # 1. Already a known node ID (fully qualified call)
    if raw_target in node_ids:
        return raw_target

    parts = source.split(".")

    # 2. Qualify with each ancestor prefix of the source
    # source="pkg.mod.Class.method", try "pkg.mod.Class.helper", "pkg.mod.helper", "pkg.helper"
    for i in range(len(parts) - 1, 0, -1):
        candidate = ".".join(parts[:i]) + "." + raw_target
        if candidate in node_ids:
            return candidate

    # 3. Handle dotted targets via imports: "EdgeKind.CALLS" where EdgeKind was imported
    first_part = raw_target.split(".")[0]
    rest = raw_target.split(".")[1:]
    for i in range(len(parts), 0, -1):
        mod_candidate = ".".join(parts[:i])
        if mod_candidate in import_map:
            if first_part in import_map[mod_candidate]:
                imported_target = import_map[mod_candidate][first_part]
                if rest:
                    full = imported_target + "." + ".".join(rest)
                else:
                    full = imported_target
                if full in node_ids:
                    return full

    return None
