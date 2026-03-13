"""
Top-level analysis orchestrator.

Runs the full structural analysis pipeline on a projected graph and produces
a unified StructuralAnalysis result with trust metadata and prioritized
findings.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from topo_parser.graph import CodeGraph, EdgeKind
from topo_analyzer.anomalies import Anomaly, AnomalyKind, detect_anomalies
from topo_analyzer.modules import Module, ModuleDetection, detect_modules
from topo_analyzer.projection import (
    AnalysisAnchor,
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

    def summary(self, verbose: bool = False) -> str:
        """Human-readable summary of structural analysis."""
        lines = [
            f"Analysis graph: {self.graph.node_count} nodes, {self.graph.edge_count} edges",
            f"Projection: {self.projection.config.level.value} level, "
            f"{', '.join(kind.value for kind in self.projection.config.edge_kinds)}, "
            f"{'internal-only' if self.projection.config.internal_only else 'mixed'}",
        ]

        if self.projection.config.scope_roots:
            lines.append(f"Scope roots: {', '.join(self.projection.config.scope_labels)}")

        if self.coverage:
            lines.append("")
            lines.append("Coverage:")
            lines.append(
                f"  Parsed graph: {self.coverage.raw_node_count} nodes, {self.coverage.raw_edge_count} edges"
            )
            lines.append(
                f"  Scope selection: {self.coverage.scoped_node_count}/{self.coverage.raw_node_count} "
                f"nodes ({self.coverage.scope_node_ratio:.1%})"
            )
            lines.append(
                f"  Projection: {self.coverage.scoped_node_count} scoped nodes -> "
                f"{self.coverage.analyzed_node_count} analysis nodes "
                f"({self.coverage.projection_node_ratio:.1%} compression ratio)"
            )
            lines.append(
                f"  Analysis graph: {self.coverage.analyzed_node_count} nodes, "
                f"{self.coverage.analyzed_edge_count} edges"
            )
            lines.append(
                f"  Spectral coverage: {self.coverage.spectral_node_count}/{self.coverage.analyzed_node_count} "
                f"nodes ({self.coverage.spectral_coverage_ratio:.1%}) across "
                f"{self.coverage.component_count} components"
            )

        lines.append("")
        lines.append("Findings:")
        if self.findings:
            for finding in self.findings[:7]:
                lines.append(
                    f"  [{finding.severity_label}/{finding.confidence_label}] {finding.title}: "
                    f"{finding.description}"
                )
                for anchor in finding.anchors[:2]:
                    lines.append(f"    {anchor.node_id} ({anchor.file}:{anchor.line})")
        else:
            lines.append("  No high-signal issues detected.")

        if self.cross_package_dependencies:
            lines.append("")
            lines.append("Package flow:")
            for dependency in self.cross_package_dependencies:
                lines.append(f"  {dependency.summary()}")

        if self.health:
            lines.append("")
            lines.append("Health:")
            lines.append(
                f"  Call density: {self.health.call_count} calls / {self.health.analyzed_node_count} "
                f"nodes = {self.health.call_density:.2f} calls/node"
            )
            lines.append(
                f"  Orphans: {self.health.orphan_count}/{self.graph.node_count} "
                f"({self.health.orphan_ratio:.1%})"
            )
            lines.append(
                f"  Largest module: {self.health.largest_module_ratio:.1%} "
                f"({self.health.largest_module_status})"
            )

        lines.append("")
        lines.append(f"Detected modules: {len(self.modules)}")
        for module in self.modules:
            share = module.size / self.graph.node_count if self.graph.node_count else 0.0
            flag = " unassigned" if module.unassigned else ""
            lines.append(
                f"  Module {module.id} ({module.size} nodes, {share:.1%}, "
                f"confidence {_confidence_label(module.confidence)}){flag}"
            )

        role_counts: dict[str, int] = {}
        for role in self.roles:
            role_counts[role.role.value] = role_counts.get(role.role.value, 0) + 1
        lines.append("")
        lines.append("Structural roles:")
        for role, count in sorted(role_counts.items()):
            lines.append(f"  {role}: {count}")

        if verbose and self.spectral:
            lines.append("")
            lines.append("Diagnostics:")
            lines.append(f"  Algebraic connectivity: {self.spectral.fiedler_value:.4f}")
            lines.append(f"  Spectral dimensions: {len(self.spectral.eigenvalues)}")
            if self.module_detection.silhouette is not None:
                lines.append(f"  Silhouette: {self.module_detection.silhouette:.3f}")
            if self.module_detection.chosen_k is not None:
                lines.append(f"  Chosen k: {self.module_detection.chosen_k}")

        return "\n".join(lines)

    def to_dict(self, *, include_raw: bool = True) -> dict:
        """Convert the full analysis result into a JSON-serializable dict."""
        return {
            "scope": {
                "level": self.projection.config.level.value,
                "edge_kinds": [kind.value for kind in self.projection.config.edge_kinds],
                "internal_only": self.projection.config.internal_only,
                "roots": self.projection.config.scope_labels,
            },
            "graph": {
                "nodes": self.graph.node_count,
                "edges": self.graph.edge_count,
            },
            "raw_graph": {
                "nodes": self.raw_graph.node_count,
                "edges": self.raw_graph.edge_count,
            } if include_raw else None,
            "coverage": {
                "raw_nodes": self.coverage.raw_node_count,
                "raw_edges": self.coverage.raw_edge_count,
                "scoped_nodes": self.coverage.scoped_node_count,
                "scoped_edges": self.coverage.scoped_edge_count,
                "analyzed_nodes": self.coverage.analyzed_node_count,
                "analyzed_edges": self.coverage.analyzed_edge_count,
                "scope_filtered_nodes": self.coverage.scope_filtered_node_count,
                "scope_filtered_edges": self.coverage.scope_filtered_edge_count,
                "scope_node_ratio": round(self.coverage.scope_node_ratio, 4),
                "projection_node_ratio": round(self.coverage.projection_node_ratio, 4),
                "spectral_nodes": self.coverage.spectral_node_count,
                "spectral_coverage_ratio": round(self.coverage.spectral_coverage_ratio, 4),
                "component_count": self.coverage.component_count,
                "clusterable_component_count": self.coverage.clusterable_component_count,
                "largest_component_ratio": round(self.coverage.largest_component_ratio, 4),
            } if self.coverage else None,
            "spectral": {
                "fiedler_value": self.spectral.fiedler_value,
                "eigenvalues": self.spectral.eigenvalues.tolist(),
                "component_count": self.spectral.component_count,
                "clusterable_component_count": self.spectral.clusterable_component_count,
                "coverage_ratio": round(self.spectral.coverage_ratio, 4),
                "largest_component_ratio": round(self.spectral.largest_component_ratio, 4),
            } if self.spectral else None,
            "clustering": {
                "module_count": len(self.modules),
                "chosen_k": self.module_detection.chosen_k,
                "silhouette": round(self.module_detection.silhouette, 4)
                if self.module_detection.silhouette is not None else None,
                "component_count": self.module_detection.component_count,
                "clustered_node_count": self.module_detection.clustered_node_count,
                "unassigned_node_count": self.module_detection.unassigned_node_count,
            },
            "findings": [
                {
                    "kind": finding.kind,
                    "title": finding.title,
                    "description": finding.description,
                    "severity": round(finding.severity, 2),
                    "severity_label": finding.severity_label,
                    "confidence": round(finding.confidence, 2),
                    "confidence_label": finding.confidence_label,
                    "anchors": [anchor.to_dict() for anchor in finding.anchors],
                }
                for finding in self.findings
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
                for module in self.modules
            ],
            "roles": [
                {
                    "node_id": role.node_id,
                    "role": role.role.value,
                    "degree": role.degree,
                    "betweenness": round(role.betweenness, 4),
                    "in_degree": role.in_degree,
                    "out_degree": role.out_degree,
                    "anchors": [anchor.to_dict() for anchor in self.projection.anchors_for([role.node_id], limit=1)],
                }
                for role in self.roles
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
                for anomaly in self.anomalies
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
                for dependency in self.cross_package_dependencies
            ],
            "health": {
                "call_count": self.health.call_count,
                "analyzed_node_count": self.health.analyzed_node_count,
                "call_density": round(self.health.call_density, 2),
                "orphan_count": self.health.orphan_count,
                "orphan_ratio": round(self.health.orphan_ratio, 4),
                "largest_module_size": self.health.largest_module_size,
                "largest_module_ratio": round(self.health.largest_module_ratio, 4),
                "largest_module_status": self.health.largest_module_status,
            } if self.health else None,
        }


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
    )


def _build_findings(
    coverage: CoverageSummary,
    health: GraphHealth,
    dependencies: list[CrossPackageDependency],
    anomalies: list[Anomaly],
) -> list[Finding]:
    """Convert low-level diagnostics into a short findings list."""
    findings: list[Finding] = []

    if coverage.spectral_coverage_ratio < 0.75:
        findings.append(Finding(
            kind="coverage",
            title="Low spectral coverage",
            description=(
                f"Only {coverage.spectral_coverage_ratio:.1%} of analyzed nodes received "
                f"spectral fingerprints; disconnected components may limit clustering quality."
            ),
            severity=max(0.3, 1.0 - coverage.spectral_coverage_ratio),
            confidence=0.9,
        ))

    if health.largest_module_ratio >= 0.5:
        findings.append(Finding(
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
            kind="reverse_dependency",
            title=f"Reverse dependency between {pair[0]} and {pair[1]}",
            description=(
                f"Dependency flow is bidirectional with {reverse}/{total} edges in the weaker direction."
            ),
            severity=min(1.0, 0.35 + reverse / max(total, 1)),
            confidence=min(1.0, 0.5 + total / 20.0),
            anchors=anchors[:3],
        ))

    for anomaly in anomalies[:5]:
        if anomaly.kind == AnomalyKind.CROSS_MODULE:
            continue
        findings.append(Finding(
            kind=anomaly.kind.value,
            title=_anomaly_title(anomaly.kind),
            description=anomaly.description,
            severity=anomaly.severity,
            confidence=anomaly.confidence,
            anchors=anomaly.anchors,
        ))

    findings.sort(key=lambda finding: (-finding.severity, -finding.confidence, finding.title))
    return findings[:7]


def _anomaly_title(kind: AnomalyKind) -> str:
    """Human-readable titles for anomaly kinds."""
    titles = {
        AnomalyKind.CROSS_MODULE: "Unexpected reverse boundary",
        AnomalyKind.SPECTRAL_OUTLIER: "Structural outlier",
        AnomalyKind.CYCLE_MEMBER: "Dependency cycle",
    }
    return titles[kind]


def analyze(
    graph: CodeGraph,
    edge_kind: EdgeKind = EdgeKind.CALLS,
    combined: bool = False,
    n_modules: int | None = None,
    projection_config: AnalysisProjectionConfig | None = None,
) -> StructuralAnalysis:
    """
    Run the full structural analysis pipeline.

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

    projection = build_projection(graph, projection_config)
    analysis_graph = projection.graph
    active_edge_kinds = list(projection_config.edge_kinds)
    use_multilayer = combined or len(active_edge_kinds) > 1

    if use_multilayer:
        spectral = spectral_decomposition_multilayer(
            analysis_graph,
            layer_weights=projection_config.layer_weights,
        )
    else:
        spectral = spectral_decomposition(analysis_graph, edge_kind=active_edge_kinds[0])

    module_detection = detect_modules(spectral, n_modules=n_modules) if spectral else ModuleDetection(
        modules=[],
        chosen_k=None,
        silhouette=None,
        component_count=0,
        clustered_node_count=0,
        unassigned_node_count=0,
    )
    roles = classify_roles(analysis_graph, edge_kinds=active_edge_kinds)
    anomalies = detect_anomalies(
        analysis_graph,
        spectral,
        module_detection.modules,
        edge_kind=active_edge_kinds[0],
        edge_kinds=active_edge_kinds,
        projection=projection,
    )
    dependencies = _collect_cross_package_dependencies(analysis_graph, projection)
    health = _compute_health(analysis_graph, roles, module_detection.modules)
    coverage = _compute_coverage(graph, projection, spectral)
    findings = _build_findings(coverage, health, dependencies, anomalies)

    return StructuralAnalysis(
        raw_graph=graph,
        graph=analysis_graph,
        projection=projection,
        spectral=spectral,
        module_detection=module_detection,
        roles=roles,
        anomalies=anomalies,
        findings=findings,
        cross_package_dependencies=dependencies,
        health=health,
        coverage=coverage,
    )
