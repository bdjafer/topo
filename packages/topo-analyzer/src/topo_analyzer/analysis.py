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

        # Build role lookup (excluding dunders)
        role_map: dict[str, RoleAssignment] = {}
        for r in self.roles:
            if not r.node_id.split(".")[-1].startswith("_"):
                role_map[r.node_id] = r

        # Build module membership
        node_to_mod: dict[str, int] = {}
        for mod in self.modules:
            for nid in mod.node_ids:
                node_to_mod[nid] = mod.id

        # Show modules with their members grouped by role
        lines.append(f"Detected modules: {len(self.modules)}")
        for mod in self.modules:
            # Compute top-level package distribution
            pkg_counts: dict[str, int] = {}
            for nid in mod.node_ids:
                pkg = nid.split(".")[0]
                pkg_counts[pkg] = pkg_counts.get(pkg, 0) + 1
            pkg_str = ", ".join(f"{p}:{c}" for p, c in sorted(pkg_counts.items(), key=lambda x: -x[1]))
            lines.append(f"  Module {mod.id} ({mod.size} entities) [{pkg_str}]")

            # Group notable members by role (skip orphan and regular)
            notable = []
            for nid in mod.node_ids:
                r = role_map.get(nid)
                if r and r.role.value not in ("orphan", "regular"):
                    notable.append(r)
            if notable:
                notable.sort(key=lambda r: (-r.degree, r.node_id))
                for r in notable[:5]:  # top 5 per module
                    lines.append(f"    {r.role.value:12s} {r.node_id}")
                if len(notable) > 5:
                    lines.append(f"    ... and {len(notable) - 5} more")

        # Role summary
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
            for a in self.anomalies[:10]:
                lines.append(f"  [{a.severity:.1f}] {a.kind.value}: {a.description}")
            if len(self.anomalies) > 10:
                lines.append(f"  ... and {len(self.anomalies) - 10} more")

        return "\n".join(lines)


def analyze(
    graph: CodeGraph,
    edge_kind: EdgeKind = EdgeKind.CALLS,
    combined: bool = False,
    n_modules: int | None = None,
) -> StructuralAnalysis:
    """
    Run the full structural analysis pipeline.

    Args:
        graph: Parsed code graph.
        edge_kind: Primary relationship layer to analyze (ignored if combined=True).
        combined: If True, use all edge layers weighted together.
        n_modules: Force a specific number of modules (auto-detected if None).

    Returns:
        Complete structural analysis.
    """
    if combined:
        spectral = spectral_decomposition_multilayer(graph)
        modules = detect_modules(spectral, n_modules=n_modules) if spectral else []
        roles = classify_roles(graph, edge_kinds=ALL_EDGE_KINDS)
        anomalies = detect_anomalies(graph, spectral, modules, edge_kind=EdgeKind.CALLS)
    else:
        spectral = spectral_decomposition(graph, edge_kind=edge_kind)
        modules = detect_modules(spectral, n_modules=n_modules) if spectral else []
        roles = classify_roles(graph, edge_kind=edge_kind)
        anomalies = detect_anomalies(graph, spectral, modules, edge_kind=edge_kind)

    return StructuralAnalysis(
        graph=graph,
        spectral=spectral,
        modules=modules,
        roles=roles,
        anomalies=anomalies,
    )
