"""
Human-readable and JSON reporting for structural analysis results.

Separated from analysis.py to isolate presentation concerns from the
analysis orchestration pipeline.
"""

from __future__ import annotations

from pathlib import Path

from topo_analyzer.analysis import (
    SUMMARY_EDGE_KINDS,
    _edge_kind_label,
    _module_label,
)
from topo_analyzer.modules import Module
from topo_analyzer.roles import RoleAssignment, StructuralRole


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
        suffix = node_id[len(module_label) + 1:]
        return suffix
    return node_id


def _role_description(role: RoleAssignment) -> str:
    """One-line description of why a node has its structural role."""
    if role.role == StructuralRole.HUB:
        return f"degree {role.degree}"
    if role.role == StructuralRole.BRIDGE:
        return f"betweenness {role.betweenness:.3f}"
    if role.role == StructuralRole.ENTRY_POINT:
        return f"{role.out_degree} outbound, {role.in_degree} inbound"
    if role.role == StructuralRole.UTILITY:
        return f"{role.in_degree} inbound, {role.out_degree} outbound"
    return ""


def _relative_path(file_path: Path, project_root: Path | None) -> str:
    """Return a relative path string if possible, otherwise absolute."""
    if project_root is not None:
        try:
            return str(file_path.relative_to(project_root))
        except ValueError:
            pass
    return str(file_path)


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
    dependencies: list,
    style: Style,
) -> list[str]:
    """Render package dependencies as a layered ASCII DAG."""
    if not dependencies:
        return []

    # Build directed graph
    pkgs: set[str] = set()
    fwd: dict[str, set[str]] = {}
    for dep in dependencies:
        pkgs.add(dep.source_package)
        pkgs.add(dep.target_package)
        fwd.setdefault(dep.source_package, set()).add(dep.target_package)
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


# ── Public reporting functions ─────────────────────────────────────


def format_summary(
    analysis,
    *,
    verbose: bool = False,
    diagnostics: bool = False,
    ignores: dict[str, str] | None = None,
    project_root: Path | None = None,
    color: bool = False,
) -> str:
    """Human-readable structural analysis report."""
    ignores = ignores or {}
    s = Style(enabled=color)
    lines: list[str] = []

    # ── Header ──────────────────────────────────────────────────
    root_label = str(project_root) if project_root else ""
    parsed_count = analysis.coverage.raw_node_count if analysis.coverage else analysis.graph.node_count
    lines.append(s.bold(f"topo \u2014 {root_label}"))
    lines.append(
        f"{analysis.graph.node_count} nodes, {analysis.graph.edge_count} edges "
        f"({parsed_count} symbols parsed)"
    )

    # ── Issues (first — actionable section) ────────────────────
    active_findings = [f for f in analysis.findings if f.id not in ignores]
    acknowledged = [f for f in analysis.findings if f.id in ignores]
    issue_count = len(active_findings)

    lines.append("")
    lines.append(_section_header(f"Issues ({issue_count})", s))
    lines.append("")

    if active_findings:
        for finding in active_findings:
            sev_tag = s.severity(finding.severity_label, f"[{finding.severity_label}]")
            lines.append(f"  {sev_tag} {s.bold(finding.id)}")
            lines.append(f"    {finding.description}")
            for anchor in finding.anchors[:1]:
                path = _relative_path(anchor.file, project_root)
                lines.append(f"    \u2192 {s.cyan(f'{path}:{anchor.line}')}")
            lines.append("")

        high = sum(1 for f in active_findings if f.severity_label == "high")
        medium = sum(1 for f in active_findings if f.severity_label == "medium")
        low = sum(1 for f in active_findings if f.severity_label == "low")
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
        for finding in acknowledged:
            justification = ignores.get(finding.id, "")
            sev_tag = s.severity(finding.severity_label, f"[{finding.severity_label}]")
            lines.append(f"  {sev_tag} {s.dim(finding.id)} {s.dim('(acknowledged)')}")
            lines.append(f"    {s.dim(finding.description)}")
            if justification:
                lines.append(f"    {s.dim(f'Reason: {justification}')}")

    # ── Architecture ────────────────────────────────────────────
    has_deps = bool(analysis.cross_package_dependencies)
    if has_deps or verbose:
        lines.append("")
        lines.append(_section_header("Architecture", s))
        lines.append("")

        if verbose:
            clustered = [m for m in analysis.modules if not m.unassigned]
            labels = [_module_label(m) for m in clustered]
            label_counts: dict[str, int] = {}
            for lbl in labels:
                label_counts[lbl] = label_counts.get(lbl, 0) + 1

            for module, label in zip(clustered, labels):
                display_label = label
                if label_counts.get(label, 0) > 1:
                    display_label = f"{label} (group {module.id})"
                lines.append(f"  {display_label} ({module.size} nodes)")
                members = [_member_display(nid, label) for nid in module.node_ids]
                if len(members) <= 6:
                    lines.append(f"    {', '.join(members)}")
                else:
                    lines.append(f"    {', '.join(members[:5])}, ...")

            unassigned = [m for m in analysis.modules if m.unassigned]
            if unassigned:
                all_unassigned = [nid for m in unassigned for nid in m.node_ids]
                lines.append(f"  (unassigned: {len(all_unassigned)} nodes)")

            lines.append("")

        if has_deps:
            lines.extend(_render_dependency_dag(analysis.cross_package_dependencies, s))

            if verbose:
                lines.append("")
                max_src = max(len(d.source_package) for d in analysis.cross_package_dependencies)
                for dep in analysis.cross_package_dependencies:
                    parts = []
                    for kind in SUMMARY_EDGE_KINDS:
                        count = dep.edge_counts.get(kind, 0)
                        if count > 0:
                            parts.append(f"{count} {_edge_kind_label(kind, count)}")
                    lines.append(
                        f"  {dep.source_package:<{max_src}} \u2500\u2500\u2192 "
                        f"{dep.target_package}    {s.dim(', '.join(parts))}"
                    )

    # ── Critical Nodes ──────────────────────────────────────────
    critical_roles = [
        r for r in analysis.roles
        if r.role not in (StructuralRole.REGULAR, StructuralRole.ORPHAN)
    ]
    if critical_roles:
        role_order = {
            StructuralRole.HUB: 0,
            StructuralRole.BRIDGE: 1,
            StructuralRole.ENTRY_POINT: 2,
            StructuralRole.UTILITY: 3,
        }
        critical_roles.sort(key=lambda r: (role_order.get(r.role, 9), -r.degree))

        if not verbose:
            shown: list[RoleAssignment] = []
            counts: dict[StructuralRole, int] = {}
            for r in critical_roles:
                counts[r.role] = counts.get(r.role, 0) + 1
                if counts[r.role] <= 2:
                    shown.append(r)
            critical_roles = shown

        lines.append("")
        lines.append(_section_header("Critical Nodes", s))
        lines.append("")
        for r in critical_roles:
            label = f"{r.role.value.upper():<12}"
            desc = _role_description(r)
            lines.append(f"  {s.bold(label)} {r.node_id:<35} {s.dim(desc)}")

    # ── Health ──────────────────────────────────────────────────
    if analysis.health:
        lines.append("")
        lines.append(_section_header("Health", s))
        lines.append("")
        module_status = (
            "package grouping" if analysis.module_detection.package_fallback
            else analysis.health.largest_module_status
        )
        lines.append(
            f"  Density: {analysis.health.call_density:.2f} calls/node    "
            f"Orphans: {analysis.health.orphan_ratio:.1%}    "
            f"Largest module: {analysis.health.largest_module_ratio:.1%} ({module_status})"
        )

    # ── Diagnostics ─────────────────────────────────────────────
    if diagnostics:
        lines.append("")
        lines.append(_section_header("Diagnostics", s))
        lines.append("")
        if analysis.coverage:
            lines.append(
                f"  Parsed: {analysis.coverage.raw_node_count} nodes, "
                f"{analysis.coverage.raw_edge_count} edges"
            )
            lines.append(
                f"  Scope: {analysis.coverage.scoped_node_count}/{analysis.coverage.raw_node_count} "
                f"({analysis.coverage.scope_node_ratio:.1%})"
            )
            lines.append(
                f"  Projection: {analysis.coverage.scoped_node_count} \u2192 "
                f"{analysis.coverage.analyzed_node_count} nodes "
                f"({analysis.coverage.projection_node_ratio:.1%} compression)"
            )
            lines.append(
                f"  Spectral: {analysis.coverage.spectral_node_count}/"
                f"{analysis.coverage.analyzed_node_count} "
                f"({analysis.coverage.spectral_coverage_ratio:.1%}) across "
                f"{analysis.coverage.component_count} components"
            )
        if analysis.spectral:
            lines.append(f"  Algebraic connectivity: {analysis.spectral.fiedler_value:.4f}")
            lines.append(f"  Spectral dimensions: {len(analysis.spectral.eigenvalues)}")
        if analysis.module_detection.silhouette is not None:
            lines.append(f"  Silhouette: {analysis.module_detection.silhouette:.3f}")
        if analysis.module_detection.chosen_k is not None:
            lines.append(f"  Chosen k: {analysis.module_detection.chosen_k}")

    return "\n".join(lines)


def to_dict(analysis, *, include_raw: bool = True) -> dict:
    """Convert the full analysis result into a JSON-serializable dict."""
    return {
        "scope": {
            "level": analysis.projection.config.level.value,
            "edge_kinds": [kind.value for kind in analysis.projection.config.edge_kinds],
            "internal_only": analysis.projection.config.internal_only,
            "roots": analysis.projection.config.scope_labels,
        },
        "graph": {
            "nodes": analysis.graph.node_count,
            "edges": analysis.graph.edge_count,
        },
        "raw_graph": {
            "nodes": analysis.raw_graph.node_count,
            "edges": analysis.raw_graph.edge_count,
        } if include_raw else None,
        "coverage": {
            "raw_nodes": analysis.coverage.raw_node_count,
            "raw_edges": analysis.coverage.raw_edge_count,
            "scoped_nodes": analysis.coverage.scoped_node_count,
            "scoped_edges": analysis.coverage.scoped_edge_count,
            "analyzed_nodes": analysis.coverage.analyzed_node_count,
            "analyzed_edges": analysis.coverage.analyzed_edge_count,
            "scope_filtered_nodes": analysis.coverage.scope_filtered_node_count,
            "scope_filtered_edges": analysis.coverage.scope_filtered_edge_count,
            "scope_node_ratio": round(analysis.coverage.scope_node_ratio, 4),
            "projection_node_ratio": round(analysis.coverage.projection_node_ratio, 4),
            "spectral_nodes": analysis.coverage.spectral_node_count,
            "spectral_coverage_ratio": round(analysis.coverage.spectral_coverage_ratio, 4),
            "component_count": analysis.coverage.component_count,
            "clusterable_component_count": analysis.coverage.clusterable_component_count,
            "largest_component_ratio": round(analysis.coverage.largest_component_ratio, 4),
        } if analysis.coverage else None,
        "spectral": {
            "fiedler_value": analysis.spectral.fiedler_value,
            "eigenvalues": analysis.spectral.eigenvalues.tolist(),
            "component_count": analysis.spectral.component_count,
            "clusterable_component_count": analysis.spectral.clusterable_component_count,
            "coverage_ratio": round(analysis.spectral.coverage_ratio, 4),
            "largest_component_ratio": round(analysis.spectral.largest_component_ratio, 4),
        } if analysis.spectral else None,
        "clustering": {
            "module_count": len(analysis.modules),
            "chosen_k": analysis.module_detection.chosen_k,
            "silhouette": round(analysis.module_detection.silhouette, 4)
            if analysis.module_detection.silhouette is not None else None,
            "component_count": analysis.module_detection.component_count,
            "clustered_node_count": analysis.module_detection.clustered_node_count,
            "unassigned_node_count": analysis.module_detection.unassigned_node_count,
            "package_fallback": analysis.module_detection.package_fallback,
        },
        "findings": [
            {
                "id": finding.id,
                "kind": finding.kind,
                "title": finding.title,
                "description": finding.description,
                "severity": round(finding.severity, 2),
                "severity_label": finding.severity_label,
                "confidence": round(finding.confidence, 2),
                "confidence_label": finding.confidence_label,
                "anchors": [anchor.to_dict() for anchor in finding.anchors],
            }
            for finding in analysis.findings
        ],
        "modules": [
            {
                "id": module.id,
                "size": module.size,
                "members": module.node_ids,
                "component_id": module.component_id,
                "cohesion": round(module.cohesion, 4) if module.cohesion is not None else None,
                "separation": round(module.separation, 4) if module.separation is not None else None,
                "confidence": round(module.confidence, 4),
                "unassigned": module.unassigned,
            }
            for module in analysis.modules
        ],
        "roles": [
            {
                "node_id": role.node_id,
                "role": role.role.value,
                "degree": role.degree,
                "betweenness": round(role.betweenness, 4),
                "in_degree": role.in_degree,
                "out_degree": role.out_degree,
                "anchors": [anchor.to_dict() for anchor in analysis.projection.anchors_for([role.node_id], limit=1)],
            }
            for role in analysis.roles
        ],
        "anomalies": [
            {
                "kind": anomaly.kind.value,
                "node_ids": anomaly.node_ids,
                "description": anomaly.description,
                "severity": round(anomaly.severity, 2),
                "confidence": round(anomaly.confidence, 2),
                "anchors": [anchor.to_dict() for anchor in anomaly.anchors],
                "edge_counts": {
                    kind.value: count
                    for kind, count in anomaly.edge_counts.items()
                },
            }
            for anomaly in analysis.anomalies
        ],
        "cross_package_dependencies": [
            {
                "source_package": dependency.source_package,
                "target_package": dependency.target_package,
                "total": dependency.total_count,
                "edge_counts": {
                    kind.value: dependency.edge_counts.get(kind, 0)
                    for kind in SUMMARY_EDGE_KINDS
                    if dependency.edge_counts.get(kind, 0) > 0
                },
                "anchors": [anchor.to_dict() for anchor in dependency.anchors],
            }
            for dependency in analysis.cross_package_dependencies
        ],
        "health": {
            "call_count": analysis.health.call_count,
            "analyzed_node_count": analysis.health.analyzed_node_count,
            "call_density": round(analysis.health.call_density, 2),
            "orphan_count": analysis.health.orphan_count,
            "orphan_ratio": round(analysis.health.orphan_ratio, 4),
            "largest_module_size": analysis.health.largest_module_size,
            "largest_module_ratio": round(analysis.health.largest_module_ratio, 4),
            "largest_module_status": analysis.health.largest_module_status,
        } if analysis.health else None,
    }
