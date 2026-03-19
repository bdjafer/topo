"""
Human-readable text formatter for structural analysis results.

Consumes analysis.json dicts (matching schemas/analysis.schema.json v3).
No dependencies on topo-analyzer or topo-parser — reads plain dicts only.
"""

from __future__ import annotations

from pathlib import Path


# ── Styling ────────────────────────────────────────────────────────


class Style:
    """ANSI terminal styling with enable/disable support."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def _wrap(self, code: str, text: str) -> str:
        if not self.enabled:
            return str(text)
        return f"\033[{code}m{text}\033[0m"

    def bold(self, text: str) -> str:
        return self._wrap("1", text)

    def dim(self, text: str) -> str:
        return self._wrap("2", text)

    def red(self, text: str) -> str:
        return self._wrap("31", text)

    def yellow(self, text: str) -> str:
        return self._wrap("33", text)

    def cyan(self, text: str) -> str:
        return self._wrap("36", text)

    def green(self, text: str) -> str:
        return self._wrap("32", text)

    def severity(self, label: str, text: str) -> str:
        """Color text by severity level."""
        if label == "high":
            return self.red(text)
        if label == "medium":
            return self.yellow(text)
        return self.dim(text)


# ── Formatting helpers ─────────────────────────────────────────────


def _section_header(title: str, style: Style | None = None) -> str:
    """Return a formatted section header line."""
    prefix = f"\u2500\u2500 {title} "
    line = prefix + "\u2500" * max(0, 60 - len(prefix))
    return style.bold(line) if style else line


def _member_display(node_id: str, module_label: str) -> str:
    """Return display name for a module member, stripping the common prefix."""
    if node_id.startswith(module_label + "."):
        return node_id[len(module_label) + 1:]
    return node_id


def _role_description(role: dict) -> str:
    """One-line description of why a node has its structural role."""
    r = role["role"]
    if r == "hub":
        return f"degree {role['degree']}"
    if r == "bridge":
        return f"betweenness {role['betweenness']:.3f}"
    if r == "entry_point":
        return f"{role['out_degree']} outbound, {role['in_degree']} inbound"
    if r == "utility":
        return f"{role['in_degree']} inbound, {role['out_degree']} outbound"
    return ""


def _relative_path(file_path: str, project_root: Path | None) -> str:
    """Return a relative path string if possible, otherwise absolute."""
    if project_root is not None:
        try:
            return str(Path(file_path).relative_to(project_root))
        except ValueError:
            pass
    return file_path


# ── ASCII DAG renderer ─────────────────────────────────────────────

_U, _D, _L, _R = 1, 2, 4, 8
_BOX = {
    _U: '│', _D: '│', _U | _D: '│',
    _L: '─', _R: '─', _L | _R: '─',
    _U | _R: '└', _U | _L: '┘',
    _D | _R: '┌', _D | _L: '┐',
    _U | _D | _R: '├', _U | _D | _L: '┤',
    _U | _L | _R: '┴', _D | _L | _R: '┬',
    _U | _D | _L | _R: '┼',
}


def _render_dependency_dag(
    dependencies: list[dict],
    module_labels: dict[int, str],
    style: Style,
) -> list[str]:
    """Render module dependencies as a layered ASCII DAG.

    dependencies: list of {"source": int, "target": int, "weight": int, ...}
    module_labels: {module_id: label_string}
    """
    if not dependencies:
        return []

    # Build directed graph using module labels as node names
    pkgs: set[str] = set()
    fwd: dict[str, set[str]] = {}
    for dep in dependencies:
        src_label = module_labels.get(dep["source"], f"module-{dep['source']}")
        tgt_label = module_labels.get(dep["target"], f"module-{dep['target']}")
        pkgs.add(src_label)
        pkgs.add(tgt_label)
        fwd.setdefault(src_label, set()).add(tgt_label)
    for p in pkgs:
        fwd.setdefault(p, set())

    # Detect cycles via DFS
    back_edges: set[tuple[str, str]] = set()
    visited: set[str] = set()
    on_stack: set[str] = set()

    def _dfs(n: str) -> None:
        visited.add(n)
        on_stack.add(n)
        for s in sorted(fwd[n]):
            if s in on_stack:
                back_edges.add((n, s))
            elif s not in visited:
                _dfs(s)
        on_stack.discard(n)

    for p in sorted(pkgs):
        if p not in visited:
            _dfs(p)

    # Build DAG without back-edges
    dag: dict[str, set[str]] = {p: set() for p in pkgs}
    rev: dict[str, set[str]] = {p: set() for p in pkgs}
    for s in pkgs:
        for t in fwd[s]:
            if (s, t) not in back_edges:
                dag[s].add(t)
                rev[t].add(s)

    # Layer assignment (longest path from roots)
    layer_of: dict[str, int] = {}

    def _layer(n: str) -> int:
        if n in layer_of:
            return layer_of[n]
        layer_of[n] = 0
        layer_of[n] = max((_layer(p) for p in rev[n]), default=-1) + 1
        return layer_of[n]

    for p in pkgs:
        _layer(p)

    n_layers = max(layer_of.values(), default=0) + 1
    layers: list[list[str]] = [[] for _ in range(n_layers)]
    for p in sorted(pkgs):
        layers[layer_of[p]].append(p)

    # Order within layers (barycenter heuristic)
    for i in range(1, n_layers):
        prev_pos = {n: j for j, n in enumerate(layers[i - 1])}

        def _bary(n: str, _pp: dict[str, int] = prev_pos) -> float:
            ps = [_pp[p] for p in rev[n] if p in _pp]
            return sum(ps) / len(ps) if ps else 0.0

        layers[i].sort(key=_bary)

    # Insert virtual nodes for skip-layer edges
    virt: set[str] = set()
    adj_edges: list[tuple[str, str]] = []
    vid = 0
    for src in list(pkgs):
        for tgt in dag[src]:
            sl, tl = layer_of[src], layer_of[tgt]
            if tl - sl == 1:
                adj_edges.append((src, tgt))
            else:
                prev = src
                for lyr in range(sl + 1, tl):
                    vn = f"\x00v{vid}"
                    vid += 1
                    virt.add(vn)
                    layer_of[vn] = lyr
                    layers[lyr].append(vn)
                    adj_edges.append((prev, vn))
                    prev = vn
                adj_edges.append((prev, tgt))

    # Assign x-positions
    GAP = 3
    node_x: dict[str, int] = {}
    width = 0
    for lr in layers:
        x = 0
        for n in lr:
            w = len(n) if n not in virt else 0
            node_x[n] = x + w // 2
            x += max(w, 1) + GAP
        width = max(width, x)

    # Render
    out: list[str] = []
    for li in range(n_layers):
        # Node name row
        row = [' '] * width
        for n in layers[li]:
            if n in virt:
                continue
            w = len(n)
            sx = node_x[n] - w // 2
            for i, ch in enumerate(n):
                if 0 <= sx + i < width:
                    row[sx + i] = ch
        text = ''.join(row).rstrip()
        if text.strip():
            out.append('  ' + text)

        if li >= n_layers - 1:
            continue

        # Edges from this layer to the next
        gap_edges = [(s, t) for s, t in adj_edges if layer_of.get(s) == li]
        if not gap_edges:
            continue

        # Drop row: │ from sources
        drop = [' '] * width
        for s, _ in gap_edges:
            x = node_x[s]
            if 0 <= x < width:
                drop[x] = '│'
        ds = ''.join(drop).rstrip()
        if ds.strip():
            out.append('  ' + ds)

        # Routing row (bitmap approach)
        flags = [0] * width
        for s, t in gap_edges:
            sx, tx = node_x[s], node_x[t]
            if sx == tx:
                if 0 <= sx < width:
                    flags[sx] |= _U | _D
            else:
                if 0 <= sx < width:
                    flags[sx] |= _U | (_R if tx > sx else _L)
                if 0 <= tx < width:
                    flags[tx] |= _D | (_L if tx > sx else _R)
                for x in range(min(sx, tx) + 1, max(sx, tx)):
                    if 0 <= x < width:
                        flags[x] |= _L | _R

        rrow = [_BOX.get(flags[x], ' ') if flags[x] else ' ' for x in range(width)]
        rs = ''.join(rrow).rstrip()
        if rs.strip():
            out.append('  ' + rs)

        # Rise row: ▼ at real targets, │ at virtual pass-throughs
        rise = [' '] * width
        for _, t in gap_edges:
            x = node_x[t]
            if 0 <= x < width:
                rise[x] = '│' if t in virt else '▼'
        ris = ''.join(rise).rstrip()
        if ris.strip():
            out.append('  ' + ris)

    # Cycle annotations
    for s, t in sorted(back_edges):
        cycle_icon = style.red("\u27f2")
        cycle_label = style.dim("(cycle)")
        out.append(f'  {cycle_icon} {s} \u2192 {t} {cycle_label}')

    return out


# ── Public formatting functions ────────────────────────────────────


def format_text(
    data: dict,
    *,
    verbose: bool = False,
    diagnostics: bool = False,
    ignores: dict[str, str] | None = None,
    project_root: Path | None = None,
    color: bool = False,
) -> str:
    """Human-readable structural analysis report from an analysis.json dict."""
    ignores = ignores or {}
    s = Style(enabled=color)
    lines: list[str] = []

    coverage = data.get("coverage", {})
    architecture = data.get("architecture", {})
    spectral = data.get("spectral")
    health = data.get("health")
    issues = data.get("issues", [])
    roles = data.get("roles", [])

    # ── Header ──────────────────────────────────────────────────
    root_label = str(project_root) if project_root else ""
    analyzed_nodes = coverage.get("analyzed_nodes", 0)
    analyzed_edges = coverage.get("analyzed_edges", 0)
    parsed_nodes = coverage.get("parsed_nodes", analyzed_nodes)
    lines.append(s.bold(f"topo \u2014 {root_label}"))
    lines.append(
        f"{analyzed_nodes} nodes, {analyzed_edges} edges "
        f"({parsed_nodes} symbols parsed)"
    )

    # ── Issues (first — actionable section) ────────────────────
    active_issues = [i for i in issues if i.get("id") not in ignores]
    acknowledged = [i for i in issues if i.get("id") in ignores]
    issue_count = len(active_issues)

    lines.append("")
    lines.append(_section_header(f"Issues ({issue_count})", s))
    lines.append("")

    if active_issues:
        for issue in active_issues:
            sev_label = issue.get("severity_label", "low")
            sev_tag = s.severity(sev_label, f"[{sev_label}]")
            lines.append(f"  {sev_tag} {s.bold(issue['id'])}")
            lines.append(f"    {issue['description']}")
            anchors = issue.get("anchors", [])
            if anchors:
                anchor = anchors[0]
                path = _relative_path(anchor.get("file", ""), project_root)
                line_num = anchor.get("line", 0)
                lines.append(f"    \u2192 {s.cyan(f'{path}:{line_num}')}")
            lines.append("")

        high = sum(1 for i in active_issues if i.get("severity_label") == "high")
        medium = sum(1 for i in active_issues if i.get("severity_label") == "medium")
        low = sum(1 for i in active_issues if i.get("severity_label") == "low")
        count_parts: list[str] = []
        if high:
            count_parts.append(s.red(f"{high} high"))
        if medium:
            count_parts.append(s.yellow(f"{medium} medium"))
        if low:
            count_parts.append(s.dim(f"{low} low"))
        count_str = ", ".join(count_parts) if count_parts else "0"
        lines.append(f"  \u2716 {issue_count} issues ({count_str})")
    else:
        lines.append(s.green("  No issues detected."))

    if acknowledged:
        lines.append(f"  {len(acknowledged)} acknowledged (use --verbose to show)")

    if verbose and acknowledged:
        lines.append("")
        for issue in acknowledged:
            justification = ignores.get(issue.get("id", ""), "")
            sev_label = issue.get("severity_label", "low")
            sev_tag = s.severity(sev_label, f"[{sev_label}]")
            lines.append(f"  {sev_tag} {s.dim(issue['id'])} {s.dim('(acknowledged)')}")
            lines.append(f"    {s.dim(issue['description'])}")
            if justification:
                lines.append(f"    {s.dim(f'Reason: {justification}')}")

    # ── Architecture ────────────────────────────────────────────
    modules = architecture.get("modules", [])
    deps = architecture.get("dependencies", [])
    has_deps = bool(deps)

    if has_deps or verbose:
        lines.append("")
        lines.append(_section_header("Architecture", s))
        lines.append("")

        if verbose:
            clustered = [m for m in modules if not m.get("unassigned", False)]
            labels = [m.get("label", f"module-{m['id']}") for m in clustered]
            label_counts: dict[str, int] = {}
            for lbl in labels:
                label_counts[lbl] = label_counts.get(lbl, 0) + 1

            for module, label in zip(clustered, labels):
                display_label = label
                if label_counts.get(label, 0) > 1:
                    display_label = f"{label} (group {module['id']})"
                lines.append(f"  {display_label} ({module['size']} nodes)")
                members = [_member_display(nid, label) for nid in module.get("members", [])]
                if len(members) <= 6:
                    lines.append(f"    {', '.join(members)}")
                else:
                    lines.append(f"    {', '.join(members[:5])}, ...")

            unassigned = [m for m in modules if m.get("unassigned", False)]
            if unassigned:
                all_unassigned = [nid for m in unassigned for nid in m.get("members", [])]
                lines.append(f"  (unassigned: {len(all_unassigned)} nodes)")

            lines.append("")

        if has_deps:
            module_labels = {m["id"]: m.get("label", f"module-{m['id']}") for m in modules}
            lines.extend(_render_dependency_dag(deps, module_labels, s))

    # ── Critical Nodes ──────────────────────────────────────────
    critical_roles = [
        r for r in roles
        if r.get("role") not in ("regular", "orphan")
    ]
    if critical_roles:
        role_order = {
            "hub": 0,
            "bridge": 1,
            "entry_point": 2,
            "utility": 3,
        }
        critical_roles.sort(key=lambda r: (role_order.get(r.get("role", ""), 9), -r.get("degree", 0)))

        if not verbose:
            shown: list[dict] = []
            counts: dict[str, int] = {}
            for r in critical_roles:
                role_name = r.get("role", "")
                counts[role_name] = counts.get(role_name, 0) + 1
                if counts[role_name] <= 2:
                    shown.append(r)
            critical_roles = shown

        lines.append("")
        lines.append(_section_header("Critical Nodes", s))
        lines.append("")
        for r in critical_roles:
            label = f"{r.get('role', '').upper():<12}"
            desc = _role_description(r)
            lines.append(f"  {s.bold(label)} {r.get('node_id', ''):<35} {s.dim(desc)}")

    # ── Health ──────────────────────────────────────────────────
    if health:
        lines.append("")
        lines.append(_section_header("Health", s))
        lines.append("")
        q = health.get("modularity_q")
        q_str = f"{q:.3f}" if q is not None else "n/a"
        lines.append(f"  Modularity Q: {q_str}")

    # ── Diagnostics ─────────────────────────────────────────────
    if diagnostics:
        lines.append("")
        lines.append(_section_header("Diagnostics", s))
        lines.append("")
        lines.append(
            f"  Parsed: {coverage.get('parsed_nodes', 0)} nodes, "
            f"{coverage.get('parsed_edges', 0)} edges"
        )
        lines.append(
            f"  Analyzed: {coverage.get('analyzed_nodes', 0)} nodes, "
            f"{coverage.get('analyzed_edges', 0)} edges"
        )
        if spectral:
            lines.append(f"  Algebraic connectivity: {spectral.get('fiedler_value', 0):.4f}")
            lines.append(f"  Spectral dimensions: {len(spectral.get('eigenvalues', []))}")
            lines.append(
                f"  Components: {spectral.get('components', 0)}    "
                f"Largest: {spectral.get('largest_component_ratio', 0):.1%}"
            )
            lines.append(
                f"  Spectral coverage: {spectral.get('nodes_covered', 0)}/"
                f"{coverage.get('analyzed_nodes', 0)} "
                f"({spectral.get('coverage_ratio', 0):.1%})"
            )
        silhouette = architecture.get("silhouette")
        if silhouette is not None:
            lines.append(f"  Silhouette: {silhouette:.3f}")

    lines.append("")
    return "\n".join(lines)
