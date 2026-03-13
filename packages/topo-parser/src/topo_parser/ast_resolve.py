"""
AST-based resolution helpers for parser fallback paths.

Keeps project traversal and PyCG integration separate from the AST-only
logic used to resolve names, calls, and inheritance edges.
"""

from __future__ import annotations

import ast

from topo_parser.graph import CodeGraph, Edge, EdgeKind, NodeKind


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
    if (
        self_name
        and class_id
        and isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == self_name
    ):
        class_short = class_id.rsplit(".", 1)[-1]
        return f"{class_short}.{node.attr}"

    if (
        class_id
        and isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "super"
    ):
        if base_name:
            return f"{base_name}.{node.attr}"
        return None

    return _resolve_name(node)


def _resolve_call_edges_ast(graph: CodeGraph) -> None:
    """AST fallback for call edge extraction when PyCG is unavailable."""
    for node_obj in list(graph.nodes.values()):
        if node_obj.kind != NodeKind.FUNCTION:
            continue
        try:
            source = node_obj.file.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(node_obj.file))
        except (SyntaxError, UnicodeDecodeError):
            continue
        func_node = _find_ast_function(tree, node_obj.line, node_obj.name)
        if func_node is None:
            continue

        class_id, base_name = _find_class_context(tree, node_obj.id)
        self_name = _self_param_name(func_node) if class_id else None

        seen_calls: set[str] = set()
        for child in ast.walk(func_node):
            if isinstance(child, ast.Call):
                callee = _resolve_call(child.func, self_name, class_id, base_name)
                if callee and callee not in seen_calls:
                    seen_calls.add(callee)
                    graph.add_edge(Edge(
                        source=node_obj.id, target=callee, kind=EdgeKind.CALLS,
                    ))

    _resolve_raw_edges(graph, {EdgeKind.CALLS})


def _find_ast_function(
    tree: ast.Module, line: int, name: str,
) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    """Find a function or method node by line number and name."""
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == name
            and node.lineno == line
        ):
            return node
    return None


def _find_class_context(
    tree: ast.Module, func_id: str,
) -> tuple[str | None, str | None]:
    """Determine the enclosing class ID and first declared base class name."""
    parts = func_id.rsplit(".", 2)
    if len(parts) < 3:
        return None, None

    parent_id = func_id.rsplit(".", 1)[0]
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            if parent_id.endswith(f".{node.name}") or parent_id == node.name:
                base_name = None
                if node.bases:
                    base_name = _resolve_name(node.bases[0])
                return parent_id, base_name
    return None, None


def _resolve_name(node: ast.expr) -> str | None:
    """Best-effort resolve an AST expression to a dotted name string."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        value = _resolve_name(node.value)
        if value:
            return f"{value}.{node.attr}"
    return None


def _resolve_inherits_edges(graph: CodeGraph) -> None:
    """Resolve raw targets in INHERITS edges to actual graph node IDs."""
    _resolve_raw_edges(graph, {EdgeKind.INHERITS})


def _resolve_raw_edges(graph: CodeGraph, kinds: set[EdgeKind]) -> None:
    """Resolve raw targets in edges of given kinds to actual graph node IDs."""
    node_ids = set(graph.nodes)

    import_map: dict[str, dict[str, str]] = {}
    for edge in graph.edges_by_kind(EdgeKind.IMPORTS):
        mod = edge.source
        target = edge.target
        short = target.rsplit(".", 1)[-1]
        import_map.setdefault(mod, {})[short] = target

    keep: list[Edge] = []
    for edge in graph.edges:
        if edge.kind not in kinds:
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
    if raw_target in node_ids:
        return raw_target

    parts = source.split(".")

    for i in range(len(parts) - 1, 0, -1):
        candidate = ".".join(parts[:i]) + "." + raw_target
        if candidate in node_ids:
            return candidate

    first_part = raw_target.split(".")[0]
    rest = raw_target.split(".")[1:]
    for i in range(len(parts), 0, -1):
        mod_candidate = ".".join(parts[:i])
        if mod_candidate in import_map and first_part in import_map[mod_candidate]:
            imported_target = import_map[mod_candidate][first_part]
            full = imported_target if not rest else imported_target + "." + ".".join(rest)
            if full in node_ids:
                return full

    return None
