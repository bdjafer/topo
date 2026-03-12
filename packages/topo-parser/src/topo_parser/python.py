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
    """
    graph = CodeGraph()
    py_files = sorted(root.rglob("*.py"))

    for py_file in py_files:
        try:
            source = py_file.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(py_file))
        except (SyntaxError, UnicodeDecodeError):
            continue

        _extract_from_file(graph, tree, py_file, root)

    return graph


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

    # Methods
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            _extract_function(graph, child, class_id, file)


def _extract_function(
    graph: CodeGraph, node: ast.FunctionDef | ast.AsyncFunctionDef, parent_id: str, file: Path
) -> None:
    """Extract a function/method node and its call edges."""
    func_id = f"{parent_id}.{node.name}"
    graph.add_node(Node(id=func_id, kind=NodeKind.FUNCTION, file=file, line=node.lineno, name=node.name))
    graph.add_edge(Edge(source=parent_id, target=func_id, kind=EdgeKind.CONTAINS))

    # Walk the function body for call expressions
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            callee = _resolve_name(child.func)
            if callee:
                graph.add_edge(Edge(source=func_id, target=callee, kind=EdgeKind.CALLS))


def _extract_import(graph: CodeGraph, node: ast.Import | ast.ImportFrom, mod_id: str) -> None:
    """Extract import edges."""
    if isinstance(node, ast.Import):
        for alias in node.names:
            graph.add_edge(Edge(source=mod_id, target=alias.name, kind=EdgeKind.IMPORTS))
    elif node.module:
        graph.add_edge(Edge(source=mod_id, target=node.module, kind=EdgeKind.IMPORTS))


def _resolve_name(node: ast.expr) -> str | None:
    """Best-effort resolve an AST expression to a dotted name string."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        value = _resolve_name(node.value)
        if value:
            return f"{value}.{node.attr}"
    return None
