"""Report formatting for experiment results."""

from __future__ import annotations

import math

from topo_benchmark.experiments.exp1_architecture import Experiment1Result


def format_exp1_report(result: Experiment1Result) -> str:
    """Format Experiment 1 results as a readable report."""
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("  EXPERIMENT 1: Cross-Directory Architecture Recovery")
    lines.append("=" * 70)
    lines.append("")

    # Per-codebase table
    lines.append("Per-Codebase Results:")
    lines.append("-" * 70)
    lines.append(
        f"{'Codebase':<12} {'V-meas(S)':<10} {'V-meas(D)':<10} {'V-meas(L)':<10} "
        f"{'XDir(S)':<10} {'XDir(L)':<10} {'Beats D?':<9} {'Beats L?':<9}"
    )
    lines.append("-" * 70)

    for cr in result.codebase_results:
        if cr.error:
            lines.append(f"{cr.name:<12} ERROR: {cr.error}")
            continue

        xdir_s = f"{cr.spectral_cross_dir_recovery:.1%}" if not math.isnan(cr.spectral_cross_dir_recovery) else "N/A"
        xdir_l = f"{cr.louvain_cross_dir_recovery:.1%}" if not math.isnan(cr.louvain_cross_dir_recovery) else "N/A"

        lines.append(
            f"{cr.name:<12} {cr.spectral_v_measure:<10.3f} {cr.directory_v_measure:<10.3f} "
            f"{cr.louvain_v_measure:<10.3f} {xdir_s:<10} {xdir_l:<10} "
            f"{'YES' if cr.spectral_beats_directory else 'no':<9} "
            f"{'YES' if cr.spectral_beats_louvain else 'no':<9}"
        )

    lines.append("")

    # Boundary F1 table
    lines.append("Boundary F1:")
    lines.append("-" * 50)
    lines.append(f"{'Codebase':<12} {'Spectral':<12} {'Directory':<12} {'Louvain':<12}")
    lines.append("-" * 50)
    for cr in result.codebase_results:
        if cr.error:
            continue
        lines.append(
            f"{cr.name:<12} {cr.spectral_boundary_f1:<12.3f} "
            f"{cr.directory_boundary_f1:<12.3f} {cr.louvain_boundary_f1:<12.3f}"
        )
    lines.append("")

    # Cluster details
    for cr in result.codebase_results:
        if cr.error or not cr.spectral_clusters:
            continue
        lines.append(f"Spectral clusters for {cr.name}:")
        for cid, members in sorted(cr.spectral_clusters.items()):
            lines.append(f"  Cluster {cid}: {', '.join(members)}")
        lines.append("")

    # Verdict
    lines.append("=" * 70)
    lines.append(f"  VERDICT: {result.verdict}")
    lines.append("=" * 70)
    for detail in result.verdict_details:
        lines.append(f"  - {detail}")
    lines.append("")

    return "\n".join(lines)
