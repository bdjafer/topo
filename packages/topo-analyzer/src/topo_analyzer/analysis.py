"""
Top-level analysis orchestrator.

Runs the full structural analysis pipeline on a CodeGraph and produces
a unified StructuralAnalysis result.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from topo_parser.graph import CodeGraph, EdgeKind
from topo_analyzer.spectral import (
    SpectralResult,
    spectral_decomposition,
    spectral_decomposition_multilayer,
)
from topo_analyzer.modules import Module, detect_modules
from topo_analyzer.roles import RoleAssignment, classify_roles
from topo_analyzer.anomalies import Anomaly, detect_anomalies

ALL_EDGE_KINDS = [EdgeKind.CALLS, EdgeKind.IMPORTS, EdgeKind.CONTAINS, EdgeKind.INHERITS]


@dataclass
class StructuralAnalysis:
    """Complete structural analysis of a codebase."""

    graph: CodeGraph
    spectral: SpectralResult | None
    modules: list[Module]
    roles: list[RoleAssignment]
    anomalies: list[Anomaly] = field(default_factory=list)

    def summary(self) -> str:
        """Human-readable summary of structural analysis."""
        lines = [self.graph.summary(), ""]

        if self.spectral:
            lines.append(f"Algebraic connectivity (Fiedler value): {self.spectral.fiedler_value:.4f}")
            lines.append(f"Spectral dimensions: {len(self.spectral.eigenvalues)}")
            lines.append("")

        lines.append(f"Detected modules: {len(self.modules)}")
        for mod in self.modules:
            lines.append(f"  Module {mod.id}: {mod.size} entities")

        lines.append("")
        role_counts: dict[str, int] = {}
        for r in self.roles:
            role_counts[r.role.value] = role_counts.get(r.role.value, 0) + 1
        lines.append("Structural roles:")
        for role, count in sorted(role_counts.items()):
            lines.append(f"  {role}: {count}")

        if self.anomalies:
            lines.append("")
            lines.append(f"Anomalies: {len(self.anomalies)}")
            for a in self.anomalies:
                lines.append(f"  [{a.severity:.1f}] {a.kind.value}: {a.description}")

        return "\n".join(lines)


def analyze(
    graph: CodeGraph,
    edge_kind: EdgeKind = EdgeKind.CALLS,
    combined: bool = False,
) -> StructuralAnalysis:
    """
    Run the full structural analysis pipeline.

    Args:
        graph: Parsed code graph.
        edge_kind: Primary relationship layer to analyze (ignored if combined=True).
        combined: If True, use all edge layers weighted together.

    Returns:
        Complete structural analysis.
    """
    if combined:
        spectral = spectral_decomposition_multilayer(graph)
        modules = detect_modules(spectral) if spectral else []
        roles = classify_roles(graph, edge_kinds=ALL_EDGE_KINDS)
        # Use CALLS layer for cycle/cross-module anomaly detection
        anomalies = detect_anomalies(graph, spectral, modules, edge_kind=EdgeKind.CALLS)
    else:
        spectral = spectral_decomposition(graph, edge_kind=edge_kind)
        modules = detect_modules(spectral) if spectral else []
        roles = classify_roles(graph, edge_kind=edge_kind)
        anomalies = detect_anomalies(graph, spectral, modules, edge_kind=edge_kind)

    return StructuralAnalysis(
        graph=graph,
        spectral=spectral,
        modules=modules,
        roles=roles,
        anomalies=anomalies,
    )
