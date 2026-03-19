"""
Top-level analysis orchestrator.

Runs the full structural analysis pipeline on a projected graph and produces
a unified StructuralAnalysis result with trust metadata and prioritized
findings.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from collections import Counter

from topo_parser.graph import CodeGraph, EdgeKind
from topo_analyzer.anomalies import Anomaly, AnomalyKind, detect_anomalies
from topo_analyzer.modules import Module, ModuleDetection, detect_modules
from topo_analyzer.projection import (
    AnalysisAnchor,
    AnalysisLevel,
    AnalysisProjection,
    AnalysisProjectionConfig,
    build_projection,
)
from topo_analyzer.roles import RoleAssignment, StructuralRole, classify_roles
from topo_analyzer.spectral import (
    SpectralResult,
    spectral_decomposition,
    spectral_decomposition_multilayer,
)
from topo_analyzer._rust_backend import is_available as _rust_available, run_core_analysis

SUMMARY_EDGE_KINDS = [EdgeKind.CALLS, EdgeKind.IMPORTS, EdgeKind.INHERITS, EdgeKind.CONTAINS]


def _edge_kind_label(kind: EdgeKind, count: int) -> str:
    """Return a human-readable label for an edge kind count."""
    singular = {
        EdgeKind.CALLS: "call",
        EdgeKind.IMPORTS: "import",
        EdgeKind.INHERITS: "inherit",
        EdgeKind.CONTAINS: "contains",
    }
    plural = {
        EdgeKind.CALLS: "calls",
        EdgeKind.IMPORTS: "imports",
        EdgeKind.INHERITS: "inherits",
        EdgeKind.CONTAINS: "contains",
    }
    return singular[kind] if count == 1 else plural[kind]


def _confidence_label(score: float) -> str:
    """Map a confidence score to a stable label."""
    if score >= 0.75:
        return "high"
    if score >= 0.45:
        return "medium"
    return "low"


def _severity_label(score: float) -> str:
    """Map a severity score to a stable label."""
    if score >= 0.75:
        return "high"
    if score >= 0.45:
        return "medium"
    return "low"


def _module_label(module: Module) -> str:
    """Derive a human-readable label from a module's members."""
    if not module.node_ids:
        return f"module-{module.id}"
    # Find the longest common dotted prefix
    parts_list = [nid.split(".") for nid in module.node_ids]
    prefix_parts: list[str] = []
    for level_parts in zip(*parts_list):
        if len(set(level_parts)) == 1:
            prefix_parts.append(level_parts[0])
        else:
            break
    if prefix_parts:
        return ".".join(prefix_parts)
    # No common prefix — use the most frequent top-level package
    top_packages = [nid.split(".", 1)[0] for nid in module.node_ids]
    most_common_pkg, _ = Counter(top_packages).most_common(1)[0]
    return most_common_pkg


@dataclass(frozen=True)
class CoverageSummary:
    """How much of the parsed graph remains after projection and spectral filtering."""

    raw_node_count: int
    raw_edge_count: int
    scoped_node_count: int
    scoped_edge_count: int
    analyzed_node_count: int
    analyzed_edge_count: int
    scope_filtered_node_count: int
    scope_filtered_edge_count: int
    scope_node_ratio: float
    projection_node_ratio: float
    spectral_node_count: int
    spectral_coverage_ratio: float
    component_count: int
    clusterable_component_count: int
    largest_component_ratio: float


@dataclass(frozen=True)
class CrossPackageDependency:
    """Aggregated dependency counts between top-level packages."""

    source_package: str
    target_package: str
    edge_counts: dict[EdgeKind, int]
    anchors: list[AnalysisAnchor] = field(default_factory=list)

    @property
    def total_count(self) -> int:
        return sum(self.edge_counts.values())

    def summary(self) -> str:
        """Render the dependency in a compact human-readable form."""
        parts = []
        for kind in SUMMARY_EDGE_KINDS:
            count = self.edge_counts.get(kind, 0)
            if count > 0:
                parts.append(f"{count} {_edge_kind_label(kind, count)}")
        return f"{self.source_package} -> {self.target_package}: {', '.join(parts)}"


@dataclass(frozen=True)
class GraphHealth:
    """High-signal health metrics for an analysis run."""

    call_count: int
    analyzed_node_count: int
    call_density: float
    orphan_count: int
    orphan_ratio: float
    largest_module_size: int
    largest_module_ratio: float
    modularity_q: float | None = None

    @property
    def largest_module_status(self) -> str:
        """Coarse interpretation of how concentrated the graph is."""
        if self.largest_module_ratio >= 0.5:
            return "poorly separated"
        if self.largest_module_ratio >= 0.35:
            return "mixed"
        return "well separated"


@dataclass(frozen=True)
class Finding:
    """A prioritized, developer-facing structural finding."""

    id: str
    kind: str
    title: str
    description: str
    severity: float
    confidence: float
    anchors: list[AnalysisAnchor] = field(default_factory=list)

    @property
    def severity_label(self) -> str:
        return _severity_label(self.severity)

    @property
    def confidence_label(self) -> str:
        return _confidence_label(self.confidence)


@dataclass
class StructuralAnalysis:
    """Complete structural analysis of a codebase."""

    raw_graph: CodeGraph
    graph: CodeGraph
    projection: AnalysisProjection
    spectral: SpectralResult | None
    module_detection: ModuleDetection
    roles: list[RoleAssignment]
    anomalies: list[Anomaly] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    cross_package_dependencies: list[CrossPackageDependency] = field(default_factory=list)
    health: GraphHealth | None = None
    coverage: CoverageSummary | None = None

    @property
    def modules(self) -> list[Module]:
        """Compatibility accessor for callers that expect direct modules."""
        return self.module_detection.modules

    def to_dict(self) -> dict:
        """JSON-serializable dict of the full analysis result.

        Delegates to :func:`topo_analyzer.report.to_dict`.
        """
        from topo_analyzer.report import to_dict
        return to_dict(self)


def _top_level_package(node_id: str) -> str:
    """Return the first path component of a node ID."""
    return node_id.split(".", 1)[0]


def _collect_cross_package_dependencies(
    graph: CodeGraph,
    projection: AnalysisProjection,
) -> list[CrossPackageDependency]:
    """Aggregate cross-package dependencies across the projected graph."""
    pair_counts: dict[tuple[str, str], dict[EdgeKind, int]] = {}
    pair_node_ids: dict[tuple[str, str], list[str]] = {}
    for edge in graph.edges:
        if edge.source not in graph.nodes or edge.target not in graph.nodes:
            continue
        src_pkg = _top_level_package(edge.source)
        tgt_pkg = _top_level_package(edge.target)
        if src_pkg == tgt_pkg:
            continue
        pair = (src_pkg, tgt_pkg)
        counts = pair_counts.setdefault(pair, {})
        counts[edge.kind] = counts.get(edge.kind, 0) + 1
        pair_node_ids.setdefault(pair, [])
        if len(pair_node_ids[pair]) < 6:
            pair_node_ids[pair].extend([edge.source, edge.target])

    dependencies = [
        CrossPackageDependency(
            source_package=source_package,
            target_package=target_package,
            edge_counts=edge_counts,
            anchors=projection.anchors_for(pair_node_ids[(source_package, target_package)]),
        )
        for (source_package, target_package), edge_counts in pair_counts.items()
    ]
    dependencies.sort(
        key=lambda dependency: (
            -dependency.total_count,
            dependency.source_package,
            dependency.target_package,
        ),
    )
    return dependencies


def _compute_coverage(
    raw_graph: CodeGraph,
    projection: AnalysisProjection,
    spectral: SpectralResult | None,
) -> CoverageSummary:
    """Compute projection and spectral coverage diagnostics."""
    return CoverageSummary(
        raw_node_count=raw_graph.node_count,
        raw_edge_count=raw_graph.edge_count,
        scoped_node_count=projection.scoped_node_count,
        scoped_edge_count=projection.scoped_edge_count,
        analyzed_node_count=projection.graph.node_count,
        analyzed_edge_count=projection.graph.edge_count,
        scope_filtered_node_count=projection.scope_filtered_node_count,
        scope_filtered_edge_count=projection.scope_filtered_edge_count,
        scope_node_ratio=projection.scope_node_ratio,
        projection_node_ratio=projection.projection_node_ratio,
        spectral_node_count=spectral.analyzed_node_count if spectral else 0,
        spectral_coverage_ratio=spectral.coverage_ratio if spectral else 0.0,
        component_count=spectral.component_count if spectral else 0,
        clusterable_component_count=spectral.clusterable_component_count if spectral else 0,
        largest_component_ratio=spectral.largest_component_ratio if spectral else 0.0,
    )


def _compute_modularity_q(graph: CodeGraph, modules: list[Module]) -> float | None:
    """Newman's modularity Q for the detected module assignment.

    Measures how well module boundaries explain the edge structure vs random
    assignment.  Range: -0.5 to 1.0.  >0.3 significant, >0.5 strong.
    """
    m = graph.edge_count
    if m == 0:
        return None

    node_to_module: dict[str, int] = {}
    for mod in modules:
        if mod.unassigned:
            continue
        for nid in mod.node_ids:
            node_to_module[nid] = mod.id

    internal: dict[int, int] = {}
    degree: dict[int, int] = {}

    for edge in graph.edges:
        src_mod = node_to_module.get(edge.source)
        tgt_mod = node_to_module.get(edge.target)
        if src_mod is not None:
            degree[src_mod] = degree.get(src_mod, 0) + 1
        if tgt_mod is not None:
            degree[tgt_mod] = degree.get(tgt_mod, 0) + 1
        if src_mod is not None and tgt_mod is not None and src_mod == tgt_mod:
            internal[src_mod] = internal.get(src_mod, 0) + 1

    q = 0.0
    for mod_id in set(internal) | set(degree):
        ec = internal.get(mod_id, 0) / m
        ac = degree.get(mod_id, 0) / (2 * m)
        q += ec - ac * ac

    return round(q, 4)


def _compute_health(
    graph: CodeGraph,
    roles: list[RoleAssignment],
    modules: list[Module],
) -> GraphHealth:
    """Compute health metrics surfaced in summaries and JSON output."""
    call_count = sum(
        1
        for edge in graph.edges_by_kind(EdgeKind.CALLS)
        if edge.source in graph.nodes and edge.target in graph.nodes
    )
    analyzed_node_count = graph.node_count
    call_density = call_count / analyzed_node_count if analyzed_node_count else 0.0
    orphan_count = sum(1 for role in roles if role.role == StructuralRole.ORPHAN)
    orphan_ratio = orphan_count / graph.node_count if graph.node_count else 0.0
    clustered_modules = [module for module in modules if not module.unassigned]
    largest_module_size = max((module.size for module in clustered_modules), default=0)
    largest_module_ratio = largest_module_size / graph.node_count if graph.node_count else 0.0

    return GraphHealth(
        call_count=call_count,
        analyzed_node_count=analyzed_node_count,
        call_density=call_density,
        orphan_count=orphan_count,
        orphan_ratio=orphan_ratio,
        largest_module_size=largest_module_size,
        largest_module_ratio=largest_module_ratio,
        modularity_q=_compute_modularity_q(graph, modules),
    )


def _issue_id(kind: str, node_ids: list[str] | None = None, packages: tuple[str, ...] | None = None) -> str:
    """Generate a stable, deterministic issue ID for a finding."""
    if kind == "coverage":
        return "coverage:low-spectral"
    if kind == "self_edge_drop":
        return "self-edge-drop:high"
    if kind == "module_separation":
        return "module-separation:weak"
    if kind == "reverse_dependency" and packages:
        return f"reverse-dependency:{','.join(sorted(packages))}"
    if kind == "spectral_outlier" and node_ids:
        return f"spectral-outlier:{node_ids[0]}"
    if kind == "cycle_member" and node_ids:
        return f"cycle:{','.join(sorted(node_ids))}"
    if kind == "orphan" and node_ids:
        return f"orphan:{','.join(sorted(node_ids))}"
    if kind == "cross_module" and node_ids:
        return f"cross-module:{','.join(sorted(node_ids))}"
    if kind == "god_module" and node_ids:
        return f"god-module:{node_ids[0]}"
    if kind == "low_cohesion" and node_ids:
        return f"low-cohesion:{node_ids[0]}"
    if kind == "fragile_hub" and node_ids:
        return f"fragile-hub:{node_ids[0]}"
    if kind == "layer_discrepancy" and node_ids:
        return f"layer-discrepancy:{node_ids[0]}"
    if kind == "wide_interface" and packages:
        return f"wide-interface:{','.join(sorted(packages))}"
    if kind == "phantom_import" and packages:
        return f"phantom-import:{','.join(sorted(packages))}"
    # Fallback
    label = node_ids[0] if node_ids else "unknown"
    return f"{kind}:{label}"


def _detect_wide_interfaces(
    graph: CodeGraph,
    modules: list[Module],
    projection: AnalysisProjection | None = None,
) -> list[Finding]:
    """Detect module pairs with unusually broad coupling surfaces.

    Counts distinct (source, target) symbol pairs crossing each module boundary.
    Flags pairs wider than 2× the median width (minimum threshold of 4).
    """
    if not modules or len(modules) < 2:
        return []

    node_to_module: dict[str, int] = {}
    module_by_id: dict[int, Module] = {}
    for m in modules:
        if m.unassigned:
            continue
        module_by_id[m.id] = m
        for nid in m.node_ids:
            node_to_module[nid] = m.id

    # Count distinct symbol pairs per module boundary.
    pair_symbols: dict[tuple[int, int], set[tuple[str, str]]] = {}
    for edge in graph.edges:
        src_mod = node_to_module.get(edge.source)
        tgt_mod = node_to_module.get(edge.target)
        if src_mod is None or tgt_mod is None or src_mod == tgt_mod:
            continue
        pair = tuple(sorted((src_mod, tgt_mod)))
        pair_symbols.setdefault(pair, set()).add((edge.source, edge.target))

    if not pair_symbols:
        return []

    widths = sorted(len(syms) for syms in pair_symbols.values())
    median_width = widths[(len(widths) - 1) // 2]
    threshold = max(2 * median_width, 4)

    findings: list[Finding] = []
    for (mod_a, mod_b), syms in pair_symbols.items():
        width = len(syms)
        if width <= threshold:
            continue
        label_a = _module_label(module_by_id[mod_a]) if mod_a in module_by_id else str(mod_a)
        label_b = _module_label(module_by_id[mod_b]) if mod_b in module_by_id else str(mod_b)
        node_ids_for_anchors = [s for pair in list(syms)[:3] for s in pair]
        findings.append(Finding(
            id=_issue_id("wide_interface", packages=(label_a, label_b)),
            kind="wide_interface",
            title=f"Wide interface: {label_a} — {label_b}",
            description=(
                f"{width} distinct coupling points between {label_a} and {label_b} "
                f"(median is {median_width}). Consider narrowing the interface."
            ),
            severity=min(1.0, 0.3 + (width - median_width) / max(width, 1) * 0.7),
            confidence=0.6,
            anchors=projection.anchors_for(node_ids_for_anchors, limit=2) if projection else [],
        ))
    return findings


def _detect_phantom_imports(
    graph: CodeGraph,
    modules: list[Module],
    projection: AnalysisProjection | None = None,
) -> list[Finding]:
    """Detect import edges between module pairs with no corresponding calls.

    An import that is never followed by a call suggests unused coupling —
    possibly a type-only import or dead dependency.
    """
    if not modules or len(modules) < 2:
        return []

    node_to_module: dict[str, int] = {}
    module_by_id: dict[int, Module] = {}
    for m in modules:
        if m.unassigned:
            continue
        module_by_id[m.id] = m
        for nid in m.node_ids:
            node_to_module[nid] = m.id

    # Collect cross-module import pairs and call pairs.
    import_pairs: dict[tuple[int, int], list[tuple[str, str]]] = {}
    call_pairs: set[tuple[int, int]] = set()

    for edge in graph.edges_by_kind(EdgeKind.IMPORTS):
        src_mod = node_to_module.get(edge.source)
        tgt_mod = node_to_module.get(edge.target)
        if src_mod is None or tgt_mod is None or src_mod == tgt_mod:
            continue
        pair = tuple(sorted((src_mod, tgt_mod)))
        import_pairs.setdefault(pair, []).append((edge.source, edge.target))

    for edge in graph.edges_by_kind(EdgeKind.CALLS):
        src_mod = node_to_module.get(edge.source)
        tgt_mod = node_to_module.get(edge.target)
        if src_mod is None or tgt_mod is None or src_mod == tgt_mod:
            continue
        call_pairs.add(tuple(sorted((src_mod, tgt_mod))))

    findings: list[Finding] = []
    for pair, examples in import_pairs.items():
        if pair in call_pairs:
            continue
        count = len(examples)
        mod_a, mod_b = pair
        label_a = _module_label(module_by_id[mod_a]) if mod_a in module_by_id else str(mod_a)
        label_b = _module_label(module_by_id[mod_b]) if mod_b in module_by_id else str(mod_b)
        node_ids = [s for src, tgt in examples[:3] for s in (src, tgt)]
        findings.append(Finding(
            id=_issue_id("phantom_import", packages=(label_a, label_b)),
            kind="phantom_import",
            title=f"Phantom import: {label_a} — {label_b}",
            description=(
                f"{count} import(s) between {label_a} and {label_b} with no "
                f"corresponding calls — possibly unused coupling or type-only imports."
            ),
            severity=min(0.5, 0.2 + count * 0.1),
            confidence=0.5,
            anchors=projection.anchors_for(node_ids, limit=2) if projection else [],
        ))
    return findings


def _build_findings(
    coverage: CoverageSummary,
    health: GraphHealth,
    dependencies: list[CrossPackageDependency],
    anomalies: list[Anomaly],
    roles: list[RoleAssignment] | None = None,
    projection: AnalysisProjection | None = None,
    package_fallback: bool = False,
    self_edge_ratio: float = 0.0,
    analysis_level: AnalysisLevel | None = None,
    graph: CodeGraph | None = None,
    modules: list[Module] | None = None,
    detail_roles: list[RoleAssignment] | None = None,
    module_detection: ModuleDetection | None = None,
) -> list[Finding]:
    """Convert low-level diagnostics into a prioritized findings list."""
    findings: list[Finding] = []

    if coverage.spectral_coverage_ratio < 0.75:
        findings.append(Finding(
            id=_issue_id("coverage"),
            kind="coverage",
            title="Low spectral coverage",
            description=(
                f"Only {coverage.spectral_coverage_ratio:.1%} of analyzed nodes received "
                f"spectral fingerprints; disconnected components may limit clustering quality."
            ),
            severity=max(0.3, 1.0 - coverage.spectral_coverage_ratio),
            confidence=0.9,
        ))

    if (
        self_edge_ratio > 0.7
        and analysis_level is not None
        and analysis_level != AnalysisLevel.SYMBOL
    ):
        findings.append(Finding(
            id=_issue_id("self_edge_drop"),
            kind="self_edge_drop",
            title="High self-edge drop rate",
            description=(
                f"{self_edge_ratio:.0%} of scoped edges collapsed into self-edges at "
                f"{analysis_level.value} level. Consider --level symbol for richer analysis."
            ),
            severity=min(1.0, 0.3 + self_edge_ratio * 0.5),
            confidence=0.9,
        ))

    # Skip module separation finding when using package fallback — the module
    # sizes reflect intentional package structure, not clustering quality.
    if health.largest_module_ratio >= 0.5 and not package_fallback:
        findings.append(Finding(
            id=_issue_id("module_separation"),
            kind="module_separation",
            title="Weak module separation",
            description=(
                f"The largest structural module still covers {health.largest_module_ratio:.1%} "
                f"of the analysis graph."
            ),
            severity=min(1.0, health.largest_module_ratio),
            confidence=0.8,
        ))

    pair_directions: dict[tuple[str, str], list[CrossPackageDependency]] = {}
    for dependency in dependencies:
        pair = tuple(sorted((dependency.source_package, dependency.target_package)))
        pair_directions.setdefault(pair, []).append(dependency)
    for pair, pair_dependencies in pair_directions.items():
        if len(pair_dependencies) < 2:
            continue
        total = sum(dependency.total_count for dependency in pair_dependencies)
        reverse = min(dependency.total_count for dependency in pair_dependencies)
        anchors: list[AnalysisAnchor] = []
        for dependency in pair_dependencies:
            anchors.extend(dependency.anchors)
        findings.append(Finding(
            id=_issue_id("reverse_dependency", packages=pair),
            kind="reverse_dependency",
            title=f"Reverse dependency between {pair[0]} and {pair[1]}",
            description=(
                f"Dependency flow is bidirectional with {reverse}/{total} edges in the weaker direction."
            ),
            severity=min(1.0, 0.35 + reverse / max(total, 1)),
            confidence=min(1.0, 0.5 + total / 20.0),
            anchors=anchors[:3],
        ))

    # Build role lookup for spectral outlier severity scaling.
    _role_weights = {
        StructuralRole.HUB: 1.5,
        StructuralRole.BRIDGE: 1.3,
        StructuralRole.ENTRY_POINT: 1.2,
        StructuralRole.UTILITY: 1.1,
    }
    detail_role_map: dict[str, StructuralRole] = {}
    if detail_roles:
        for r in detail_roles:
            detail_role_map[r.node_id] = r.role

    # Suppress spectral outlier findings when clustering quality is too low
    # for outlier detection to be informative.  Mirrors the package_fallback
    # pattern used for module-separation above.
    suppress_outliers = (
        health.largest_module_ratio >= 0.8
        or (module_detection is not None
            and module_detection.silhouette is not None
            and module_detection.silhouette < 0.3)
    )

    # Build role lookup for entry-point filtering of layer discrepancies.
    report_role_map: dict[str, StructuralRole] = {}
    if roles:
        for r in roles:
            report_role_map[r.node_id] = r.role

    for anomaly in anomalies:
        if anomaly.kind == AnomalyKind.CROSS_MODULE:
            continue
        if anomaly.kind == AnomalyKind.SPECTRAL_OUTLIER and suppress_outliers:
            continue
        # Entry points naturally span layers (high calls-out, low imports-in).
        if anomaly.kind == AnomalyKind.LAYER_DISCREPANCY and anomaly.node_ids:
            nid = anomaly.node_ids[0]
            node_role = report_role_map.get(nid) or detail_role_map.get(nid)
            if node_role == StructuralRole.ENTRY_POINT:
                continue
        severity = anomaly.severity
        if anomaly.kind == AnomalyKind.SPECTRAL_OUTLIER and anomaly.node_ids:
            role = detail_role_map.get(anomaly.node_ids[0])
            if role:
                severity = min(1.0, severity * _role_weights.get(role, 1.0))
        findings.append(Finding(
            id=_issue_id(anomaly.kind.value, node_ids=anomaly.node_ids),
            kind=anomaly.kind.value,
            title=_anomaly_title(anomaly.kind),
            description=anomaly.description,
            severity=severity,
            confidence=anomaly.confidence,
            anchors=anomaly.anchors,
        ))

    # Orphan findings — structurally disconnected nodes, possible dead code.
    if roles:
        orphan_roles = [r for r in roles if r.role == StructuralRole.ORPHAN]
        if len(orphan_roles) > 3:
            node_ids = [r.node_id for r in orphan_roles]
            anchors = projection.anchors_for(node_ids, limit=3) if projection else []
            names = ", ".join(node_ids[:4])
            if len(node_ids) > 4:
                names += ", ..."
            findings.append(Finding(
                id=_issue_id("orphan", node_ids=node_ids),
                kind="orphan",
                title=f"{len(orphan_roles)} orphan modules — possible dead code",
                description=names,
                severity=min(0.5, len(orphan_roles) * 0.1),
                confidence=0.7,
                anchors=anchors,
            ))
        else:
            for role in orphan_roles:
                anchors = projection.anchors_for([role.node_id], limit=1) if projection else []
                findings.append(Finding(
                    id=_issue_id("orphan", node_ids=[role.node_id]),
                    kind="orphan",
                    title=f"Orphan: {role.node_id}",
                    description="No inbound or outbound edges — may be dead code",
                    severity=0.3,
                    confidence=0.7,
                    anchors=anchors,
                ))

    # God module detection: modules with outsized edge share or size.
    # Suppress when clustering quality is too low — a "god module" that
    # contains everything is a clustering failure, not a code problem.
    if graph is not None and modules and not suppress_outliers:
        clustered = [m for m in modules if not m.unassigned]
        if len(clustered) >= 2:
            module_sizes = sorted(m.size for m in clustered)
            median_size = module_sizes[(len(module_sizes) - 1) // 2]
            total_edges = graph.edge_count
            k = len(clustered)
            fair_share = 2.0 / k if k > 0 else 1.0

            for m in clustered:
                label = _module_label(m)
                member_set = set(m.node_ids)
                edge_count = sum(
                    1 for e in graph.edges
                    if e.source in member_set or e.target in member_set
                )
                edge_share = edge_count / max(total_edges, 1)
                if edge_share > 2 * fair_share or m.size > 3 * median_size:
                    findings.append(Finding(
                        id=_issue_id("god_module", node_ids=[label]),
                        kind="god_module",
                        title=f"God module: {label}",
                        description=(
                            f"Module {label} has {m.size} nodes ({edge_share:.0%} of edges). "
                            f"Consider splitting it into smaller, focused modules."
                        ),
                        severity=min(1.0, 0.4 + edge_share),
                        confidence=0.7,
                        anchors=projection.anchors_for(m.node_ids[:3], limit=2) if projection else [],
                    ))

    # Low cohesion detection: modules where internal spread exceeds distinctness.
    if modules:
        for m in modules:
            if m.unassigned or m.size < 4:
                continue
            if m.cohesion is None or m.separation is None:
                continue
            if m.separation <= 0:
                continue
            if m.cohesion > m.separation:
                label = _module_label(m)
                ratio = (m.cohesion - m.separation) / m.cohesion
                findings.append(Finding(
                    id=_issue_id("low_cohesion", node_ids=[label]),
                    kind="low_cohesion",
                    title=f"Low cohesion: {label}",
                    description=(
                        f"Module {label} has higher internal spread ({m.cohesion:.3f}) "
                        f"than distinctness ({m.separation:.3f}) — members may belong "
                        f"to different concerns."
                    ),
                    severity=min(1.0, 0.3 + ratio * 0.7),
                    confidence=m.confidence,
                    anchors=projection.anchors_for(m.node_ids[:3], limit=2) if projection else [],
                ))

    # Fragile hub detection: HUB nodes that are also high-betweenness bottlenecks.
    if roles:
        hub_roles = [r for r in roles if r.role == StructuralRole.HUB]
        if hub_roles:
            all_btw = [r.betweenness for r in roles if r.betweenness > 0]
            if all_btw:
                btw_90 = sorted(all_btw)[min(int(len(all_btw) * 0.9), len(all_btw) - 1)]
                for r in hub_roles:
                    if r.betweenness >= btw_90:
                        deg_pct = r.degree / max(r.degree for rr in roles) if roles else 0
                        btw_pct = r.betweenness / max(all_btw) if all_btw else 0
                        findings.append(Finding(
                            id=_issue_id("fragile_hub", node_ids=[r.node_id]),
                            kind="fragile_hub",
                            title=f"Fragile hub: {r.node_id}",
                            description=(
                                f"{r.node_id} is both a structural hub (degree {r.degree}) "
                                f"and a high-betweenness bottleneck — single point of failure."
                            ),
                            severity=min(1.0, 0.5 + btw_pct * deg_pct * 0.5),
                            confidence=0.8,
                            anchors=projection.anchors_for([r.node_id], limit=1) if projection else [],
                        ))

    findings.sort(key=lambda finding: (-finding.severity, -finding.confidence, finding.title))
    return findings


def _anomaly_title(kind: AnomalyKind) -> str:
    """Human-readable titles for anomaly kinds."""
    titles = {
        AnomalyKind.CROSS_MODULE: "Unexpected reverse boundary",
        AnomalyKind.SPECTRAL_OUTLIER: "Structural outlier",
        AnomalyKind.CYCLE_MEMBER: "Dependency cycle",
        AnomalyKind.LAYER_DISCREPANCY: "Cross-layer discrepancy",
    }
    return titles[kind]


def _build_dual_projection(
    graph: CodeGraph,
    config: AnalysisProjectionConfig,
) -> tuple[AnalysisProjection, AnalysisProjection, dict[str, str], bool]:
    """Build detail (SYMBOL) and report (user-level) projections.

    Returns:
        (detail, report, symbol_to_report, is_same)
    """
    if config.level == AnalysisLevel.SYMBOL:
        projection = build_projection(graph, config)
        identity = {nid: nid for nid in projection.graph.nodes}
        return projection, projection, identity, True

    detail_config = AnalysisProjectionConfig(
        level=AnalysisLevel.SYMBOL,
        edge_kinds=config.edge_kinds,
        layer_weights=config.layer_weights,
        scope_roots=config.scope_roots,
        internal_only=config.internal_only,
        source_node_kinds=config.source_node_kinds,
    )
    detail = build_projection(graph, detail_config)
    report = build_projection(graph, config)

    # Map SYMBOL IDs to report-level IDs via shared raw node IDs
    symbol_to_report: dict[str, str] = {}
    for raw_id, symbol_id in detail.raw_to_projected.items():
        if raw_id in report.raw_to_projected:
            report_id = report.raw_to_projected[raw_id]
            symbol_to_report[symbol_id] = report_id
    return detail, report, symbol_to_report, False


_ROLE_PRIORITY = {
    StructuralRole.HUB: 0,
    StructuralRole.BRIDGE: 1,
    StructuralRole.ENTRY_POINT: 2,
    StructuralRole.UTILITY: 3,
    StructuralRole.ORPHAN: 4,
    StructuralRole.REGULAR: 5,
}


def _aggregate_roles_to_report_level(
    symbol_roles: list[RoleAssignment],
    symbol_to_report: dict[str, str],
) -> list[RoleAssignment]:
    """Pick the most structurally significant role per report-level node.

    ORPHAN requires unanimity: a report-level node is ORPHAN only if every
    symbol-level child is ORPHAN.  If any child is reachable, the module
    is not dead code.
    """
    children: dict[str, list[RoleAssignment]] = {}
    for role in symbol_roles:
        report_id = symbol_to_report.get(role.node_id)
        if report_id is None:
            continue
        children.setdefault(report_id, []).append(role)

    result: list[RoleAssignment] = []
    for report_id, child_roles in children.items():
        all_orphan = all(r.role == StructuralRole.ORPHAN for r in child_roles)

        chosen: RoleAssignment | None = None
        for role in child_roles:
            if role.role == StructuralRole.ORPHAN and not all_orphan:
                continue  # Skip non-unanimous ORPHAN
            if chosen is None or _ROLE_PRIORITY.get(role.role, 9) < _ROLE_PRIORITY.get(chosen.role, 9):
                chosen = role

        if chosen is None:
            chosen = child_roles[0]

        result.append(RoleAssignment(
            node_id=report_id,
            role=chosen.role,
            degree=max(r.degree for r in child_roles),
            betweenness=max(r.betweenness for r in child_roles),
            in_degree=max(r.in_degree for r in child_roles),
            out_degree=max(r.out_degree for r in child_roles),
        ))
    return result


def _aggregate_modules_to_report_level(
    modules: list[Module],
    symbol_to_report: dict[str, str],
) -> list[Module]:
    """Map module members from SYMBOL to report-level IDs, deduplicating."""
    result: list[Module] = []
    for module in modules:
        report_ids = sorted(set(
            symbol_to_report[nid]
            for nid in module.node_ids
            if nid in symbol_to_report
        ))
        if not report_ids and not module.node_ids:
            report_ids = []
        elif not report_ids:
            # Fallback: keep original IDs if no mapping exists
            report_ids = list(module.node_ids)
        result.append(Module(
            id=module.id,
            node_ids=report_ids,
            component_id=module.component_id,
            cohesion=module.cohesion,
            separation=module.separation,
            confidence=module.confidence,
            unassigned=module.unassigned,
        ))
    return result


def analyze(
    graph: CodeGraph,
    edge_kind: EdgeKind = EdgeKind.CALLS,
    combined: bool = False,
    n_modules: int | None = None,
    projection_config: AnalysisProjectionConfig | None = None,
) -> StructuralAnalysis:
    """
    Run the full structural analysis pipeline.

    Internally runs spectral decomposition, role classification, and anomaly
    detection at SYMBOL level for maximum information, then aggregates to
    the user's requested level for reporting.

    Args:
        graph: Parsed code graph.
        edge_kind: Primary relationship layer to analyze when no projection config is given.
        combined: If True, analyze multiple relationship layers together.
        n_modules: Force a specific number of modules (auto-detected if None).
        projection_config: Optional explicit analysis projection.

    Returns:
        Complete structural analysis.
    """
    if projection_config is None:
        projection_config = AnalysisProjectionConfig.for_analysis(
            edge_kind=edge_kind,
            combined=combined,
        )

    detail, report, symbol_to_report, is_same = _build_dual_projection(
        graph, projection_config,
    )
    detail_graph = detail.graph
    report_graph = report.graph
    active_edge_kinds = list(projection_config.edge_kinds)
    use_multilayer = combined or len(active_edge_kinds) > 1

    # --- Compute core: Rust backend or Python fallback ---
    _rust_spectral = None
    _rust_modules = None
    _rust_betweenness = None
    _rust_sccs = None

    if _rust_available():
        try:
            _rust_spectral, _rust_modules, _rust_betweenness, _rust_sccs = run_core_analysis(
                detail_graph,
                edge_kind=active_edge_kinds[0],
                combined=use_multilayer,
                layer_weights=projection_config.layer_weights,
                n_modules=n_modules,
            )
        except Exception:
            pass  # Fall back to Python path

    # Spectral decomposition on the DETAIL (symbol-level) graph
    if _rust_spectral is not None:
        spectral = _rust_spectral
    elif use_multilayer:
        spectral = spectral_decomposition_multilayer(
            detail_graph,
            layer_weights=projection_config.layer_weights,
        )
    else:
        spectral = spectral_decomposition(detail_graph, edge_kind=active_edge_kinds[0])

    # Module detection on DETAIL spectral result
    if _rust_modules is not None:
        module_detection = _rust_modules
    elif spectral:
        module_detection = detect_modules(spectral, n_modules=n_modules)
    else:
        module_detection = ModuleDetection(
            modules=[],
            chosen_k=None,
            silhouette=None,
            component_count=0,
            clustered_node_count=0,
            unassigned_node_count=0,
        )

    # Roles on DETAIL graph
    detail_roles = classify_roles(
        detail_graph,
        edge_kinds=active_edge_kinds,
        betweenness_override=_rust_betweenness,
    )

    # Anomalies on DETAIL graph
    anomalies = detect_anomalies(
        detail_graph,
        spectral,
        module_detection.modules,
        edge_kind=active_edge_kinds[0],
        edge_kinds=active_edge_kinds,
        projection=detail,
        sccs_override=_rust_sccs,
    )

    # Cross-package dependencies on REPORT graph
    dependencies = _collect_cross_package_dependencies(report_graph, report)

    # Aggregate to report level
    if is_same:
        report_roles = detail_roles
        report_modules = module_detection.modules
    else:
        report_roles = _aggregate_roles_to_report_level(detail_roles, symbol_to_report)
        report_modules = _aggregate_modules_to_report_level(
            module_detection.modules, symbol_to_report,
        )

    report_module_detection = ModuleDetection(
        modules=report_modules,
        chosen_k=module_detection.chosen_k,
        silhouette=module_detection.silhouette,
        component_count=module_detection.component_count,
        clustered_node_count=module_detection.clustered_node_count,
        unassigned_node_count=module_detection.unassigned_node_count,
        package_fallback=module_detection.package_fallback,
    )

    health = _compute_health(report_graph, report_roles, report_modules)
    coverage = _compute_coverage(graph, report, spectral)
    findings = _build_findings(
        coverage, health, dependencies, anomalies,
        roles=report_roles, projection=report,
        package_fallback=module_detection.package_fallback,
        self_edge_ratio=report.self_edge_ratio,
        analysis_level=projection_config.level,
        graph=report_graph,
        modules=report_modules,
        detail_roles=detail_roles,
        module_detection=module_detection,
    )

    # Wide interface detection on detail graph (symbol-level coupling points).
    findings.extend(_detect_wide_interfaces(
        detail_graph, module_detection.modules, detail,
    ))

    # Phantom import detection on detail graph (imports without calls).
    findings.extend(_detect_phantom_imports(
        detail_graph, module_detection.modules, detail,
    ))

    findings.sort(key=lambda f: (-f.severity, -f.confidence, f.title))

    return StructuralAnalysis(
        raw_graph=graph,
        graph=report_graph,
        projection=report,
        spectral=spectral,
        module_detection=report_module_detection,
        roles=report_roles,
        anomalies=anomalies,
        findings=findings,
        cross_package_dependencies=dependencies,
        health=health,
        coverage=coverage,
    )
