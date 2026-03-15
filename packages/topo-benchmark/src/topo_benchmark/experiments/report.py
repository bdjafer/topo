"""Report formatting for experiment results."""

from __future__ import annotations

import math

from topo_benchmark.experiments.exp1_architecture import Experiment1Result


def format_exp1_report(result: Experiment1Result) -> str:
    """Format Experiment 1 results as a readable report."""
    lines: list[str] = []
    lines.append("=" * 78)
    lines.append("  EXPERIMENT 1: Comprehensive Structural Health Evaluation")
    lines.append("=" * 78)
    lines.append("")

    # ---- Per-codebase summary ----
    lines.append("Per-Codebase Summary:")
    lines.append("-" * 78)
    lines.append(
        f"{'Codebase':<12} {'Best Config':<22} {'V(best)':>7} "
        f"{'V(dir)':>7} {'V(louv)':>7} {'Beats D?':>8} {'Beats L?':>8}"
    )
    lines.append("-" * 78)

    for cr in result.codebase_results:
        if cr.error:
            lines.append(f"{cr.name:<12} ERROR: {cr.error}")
            continue
        beats_d = "YES" if cr.spectral_beats_directory else "no"
        beats_l = "YES" if cr.spectral_beats_louvain else "no"
        lines.append(
            f"{cr.name:<12} {cr.best_config:<22} {cr.best_v_measure:>7.3f} "
            f"{cr.directory_v_measure:>7.3f} {cr.louvain_v_measure:>7.3f} "
            f"{beats_d:>8} {beats_l:>8}"
        )
    lines.append("")

    # ---- Configuration matrix ----
    for cr in result.codebase_results:
        if cr.error:
            continue
        lines.append(f"Configuration Matrix: {cr.name}")
        lines.append("-" * 78)
        lines.append(
            f"  {'Config':<22} {'V-meas':>7} {'Homo':>7} {'Comp':>7} "
            f"{'BndF1':>7} {'Dens':>7} {'Mods':>5} {'Mega%':>6} {'Purity':>7}"
        )
        lines.append(
            f"  {'-'*22} {'-'*7} {'-'*7} {'-'*7} "
            f"{'-'*7} {'-'*7} {'-'*5} {'-'*6} {'-'*7}"
        )
        for cfg in cr.config_results:
            if cfg.error:
                lines.append(f"  {cfg.config_name:<22} ERROR: {cfg.error[:40]}")
                continue
            degen = " !" if cfg.degenerate else "  "
            lines.append(
                f"  {cfg.config_name:<22} {cfg.v_measure:>7.3f} "
                f"{cfg.homogeneity:>7.3f} {cfg.completeness:>7.3f} "
                f"{cfg.boundary_f1:>7.3f} {cfg.density:>6.3f}{degen} "
                f"{cfg.n_modules:>5} {cfg.mega_cluster_ratio:>5.0%} "
                f"{cfg.weighted_purity:>7.3f}"
            )
        lines.append(f"  {'[baselines]':<22}")
        lines.append(
            f"  {'directory':<22} {cr.directory_v_measure:>7.3f} "
            f"{'':>7} {'':>7} {cr.directory_boundary_f1:>7.3f}"
        )
        lines.append(
            f"  {'louvain':<22} {cr.louvain_v_measure:>7.3f} "
            f"{'':>7} {'':>7} {cr.louvain_boundary_f1:>7.3f}"
        )
        lines.append("")

    # ---- Per-layer signal comparison ----
    lines.append("Per-Layer Signal Comparison (symbol level):")
    lines.append("-" * 60)
    # Collect all layer names across codebases
    all_layers: set[str] = set()
    for cr in result.codebase_results:
        if not cr.error:
            all_layers.update(cr.layer_signals.keys())
    sorted_layers = sorted(all_layers)

    if sorted_layers:
        header = f"  {'Codebase':<12}" + "".join(f" {l:>12}" for l in sorted_layers)
        lines.append(header)
        lines.append(f"  {'-'*12}" + "".join(f" {'-'*12}" for _ in sorted_layers))
        for cr in result.codebase_results:
            if cr.error:
                continue
            parts = [f"  {cr.name:<12}"]
            for layer in sorted_layers:
                v = cr.layer_signals.get(layer)
                if v is not None:
                    best = layer == max(cr.layer_signals, key=cr.layer_signals.get)
                    marker = " *" if best else "  "
                    parts.append(f" {v:>10.3f}{marker}")
                else:
                    parts.append(f" {'':>12}")
            lines.append("".join(parts))
        lines.append("  (* = best layer)")
    lines.append("")

    # ---- Cross-directory recovery ----
    lines.append("Cross-Directory Recovery:")
    lines.append("-" * 50)
    lines.append(f"  {'Codebase':<12} {'Spectral':>12} {'Louvain':>12}")
    lines.append(f"  {'-'*12} {'-'*12} {'-'*12}")
    for cr in result.codebase_results:
        if cr.error:
            continue
        xdir_s = f"{cr.spectral_cross_dir_recovery:.1%}" if not math.isnan(cr.spectral_cross_dir_recovery) else "N/A"
        xdir_l = f"{cr.louvain_cross_dir_recovery:.1%}" if not math.isnan(cr.louvain_cross_dir_recovery) else "N/A"
        lines.append(f"  {cr.name:<12} {xdir_s:>12} {xdir_l:>12}")
    lines.append("")

    # ---- Cluster purity (from best config) ----
    for cr in result.codebase_results:
        if cr.error or not cr.cluster_details:
            continue
        lines.append(f"Cluster Purity ({cr.name}, best config: {cr.best_config}):")
        lines.append("-" * 60)
        lines.append(f"  {'Cluster':>8} {'Size':>6} {'Purity':>7} {'Dominant Group':<20}")
        lines.append(f"  {'-'*8} {'-'*6} {'-'*7} {'-'*20}")
        for cd in cr.cluster_details[:15]:
            mega = " << mega" if (
                cd.size == max(d.size for d in cr.cluster_details) and cd.purity < 0.5
            ) else ""
            lines.append(
                f"  {cd.cluster_id:>8} {cd.size:>6} {cd.purity:>6.0%} "
                f"{cd.dominant_group:<20}{mega}"
            )
        if len(cr.cluster_details) > 15:
            lines.append(f"  ... and {len(cr.cluster_details) - 15} more clusters")
        lines.append("")

    # ---- Structural health ----
    lines.append("Structural Health (from best config):")
    lines.append("-" * 78)
    for cr in result.codebase_results:
        if cr.error:
            continue
        best_cfg = next(
            (c for c in cr.config_results if c.config_name == cr.best_config), None
        )
        if not best_cfg or best_cfg.error:
            continue

        lines.append(f"  {cr.name}:")

        # Roles
        if best_cfg.role_counts:
            role_parts = [
                f"{count} {role}"
                for role, count in sorted(best_cfg.role_counts.items(), key=lambda x: -x[1])
                if role != "regular"
            ]
            regular = best_cfg.role_counts.get("regular", 0)
            lines.append(f"    Roles: {', '.join(role_parts)}, {regular} regular")

        # Anomalies
        if best_cfg.anomaly_counts:
            anom_parts = [f"{count} {kind}" for kind, count in best_cfg.anomaly_counts.items()]
            lines.append(f"    Anomalies: {', '.join(anom_parts)}")
            for desc in best_cfg.top_anomalies:
                lines.append(f"      {desc}")

        # Spectral
        silh = f"{best_cfg.silhouette:.3f}" if best_cfg.silhouette is not None else "N/A"
        lines.append(
            f"    Fiedler: {best_cfg.fiedler_value:.4f}  "
            f"Silhouette: {silh}  "
            f"Modules: {best_cfg.n_modules}"
        )
        lines.append(
            f"    Density: {best_cfg.call_density:.2f} calls/node  "
            f"Orphans: {best_cfg.orphan_ratio:.1%}  "
            f"Module sep: {best_cfg.largest_module_status}"
        )
        lines.append("")

    # ---- Degeneracy flags ----
    degen_found = False
    for cr in result.codebase_results:
        if cr.error:
            continue
        for cfg in cr.config_results:
            if cfg.degenerate and not cfg.error:
                if not degen_found:
                    lines.append("Degeneracy Flags:")
                    lines.append("-" * 60)
                    degen_found = True
                lines.append(
                    f"  {cr.name} / {cfg.config_name}: "
                    f"density={cfg.density:.3f}, {cfg.graph_nodes} nodes, "
                    f"{cfg.graph_edges} edges"
                )
    if degen_found:
        lines.append("")

    # ---- Verdict ----
    lines.append("=" * 78)
    lines.append(f"  VERDICT: {result.verdict}")
    lines.append("=" * 78)
    for detail in result.verdict_details:
        lines.append(f"  - {detail}")
    lines.append("")

    return "\n".join(lines)
