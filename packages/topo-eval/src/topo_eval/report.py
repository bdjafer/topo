"""Evaluation report generation (JSON + human-readable)."""

import json
from datetime import datetime, timezone
from pathlib import Path


def generate_report(
    tier1: dict,
    tier2: dict,
    tier3: dict,
    tier4: dict,
    baselines: dict,
    gate: dict,
    model_info: dict | None = None,
    output_dir: Path | None = None,
) -> str:
    """Generate a comprehensive evaluation report.

    Args:
        tier1-4: Results from each evaluation tier.
        baselines: Baseline comparison results.
        gate: Go/no-go gate decision.
        model_info: Optional metadata about the model.
        output_dir: If provided, saves JSON and text reports here.

    Returns:
        Human-readable report string.
    """
    report_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_info": model_info or {},
        "tier1": _strip_per_graph(tier1),
        "tier2": _strip_per_repo(tier2),
        "tier3": _strip_per_repo(tier3),
        "tier4": _strip_per_repo(tier4),
        "baselines": baselines,
        "gate": gate,
    }

    # Save JSON
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / "evaluation_report.json"
        with open(json_path, "w") as f:
            json.dump(report_data, f, indent=2, default=str)

    # Generate human-readable text
    lines = []
    lines.append("=" * 70)
    lines.append("R-GIN EVALUATION REPORT")
    lines.append("=" * 70)
    lines.append(f"Timestamp: {report_data['timestamp']}")
    if model_info:
        lines.append(f"Model: {model_info.get('checkpoint', 'unknown')}")
        lines.append(f"Params: {model_info.get('n_params', '?'):,}")
    lines.append("")

    # Tier 1
    lines.append("-" * 70)
    lines.append("TIER 1: INTRINSIC METRICS")
    lines.append("-" * 70)
    lines.append(f"  Recon cosine sim:  {tier1.get('recon_cosine_sim', 0):.4f}  (target > 0.6)  {'PASS' if tier1.get('passes', {}).get('recon_cosine_sim') else 'FAIL'}")
    lines.append(f"  Cross-layer AUC:   {tier1.get('crosslayer_auc', 0):.4f}  (target > 0.75) {'PASS' if tier1.get('passes', {}).get('crosslayer_auc') else 'FAIL'}")
    lines.append(f"  R asymmetry:       {tier1.get('R_asymmetry', 0):.4f}  (target > 0.1)  {'PASS' if tier1.get('passes', {}).get('R_asymmetry') else 'FAIL'}")
    lines.append(f"  Overall: {'PASS' if tier1.get('tier1_pass') else 'FAIL'}")
    lines.append("")

    # Tier 2
    lines.append("-" * 70)
    lines.append("TIER 2: PHASE 2 AGREEMENT")
    lines.append("-" * 70)
    lines.append(f"  Rank correlation:  {tier2.get('rank_correlation_mean', 0):.4f} +/- {tier2.get('rank_correlation_std', 0):.4f}  (target > 0.3)  {'PASS' if tier2.get('passes', {}).get('rank_correlation_mean') else 'FAIL'}")
    lines.append(f"  Top-k overlap:     {tier2.get('topk_overlap_mean', 0):.4f} +/- {tier2.get('topk_overlap_std', 0):.4f}  (target > 0.3)  {'PASS' if tier2.get('passes', {}).get('topk_overlap_mean') else 'FAIL'}")
    if "per_repo" in tier2:
        for repo, stats in tier2["per_repo"].items():
            lines.append(f"    {repo}: rho={stats['rank_correlation']:.3f}  topk={stats['topk_overlap']:.3f}  n={stats['n_nodes']}")
    lines.append(f"  Overall: {'PASS' if tier2.get('tier2_pass') else 'FAIL'}")
    lines.append("")

    # Tier 3
    lines.append("-" * 70)
    lines.append("TIER 3: SYNTHETIC PERTURBATION")
    lines.append("-" * 70)
    lines.append(f"  Sensitivity:       {tier3.get('perturbation_sensitivity_mean', 0):.4f} +/- {tier3.get('perturbation_sensitivity_std', 0):.4f}  (target > 0.6)  {'PASS' if tier3.get('passes', {}).get('sensitivity') else 'FAIL'}")
    lines.append(f"  Specificity:       {tier3.get('perturbation_specificity_mean', 0):.4f} +/- {tier3.get('perturbation_specificity_std', 0):.4f}  (target > 0.75) {'PASS' if tier3.get('passes', {}).get('specificity') else 'FAIL'}")
    lines.append(f"  Precision:         {tier3.get('perturbation_precision_mean', 0):.4f} +/- {tier3.get('perturbation_precision_std', 0):.4f}")
    lines.append(f"  Error delta:       {tier3.get('perturbation_error_delta_mean', 0):.4f} +/- {tier3.get('perturbation_error_delta_std', 0):.4f}  (target > 0.05) {'PASS' if tier3.get('passes', {}).get('error_delta') else 'FAIL'}")
    lines.append(f"  Control sens:      {tier3.get('control_sensitivity_mean', 0):.4f} +/- {tier3.get('control_sensitivity_std', 0):.4f}  (target ~ 0.20) {'PASS' if tier3.get('passes', {}).get('control') else 'FAIL'}")
    lines.append(f"  Trials:            {tier3.get('n_trials_total', 0)} across {tier3.get('n_repos_evaluated', 0)} repos")
    lines.append(f"  Overall: {'PASS' if tier3.get('tier3_pass') else 'FAIL'}")
    lines.append("")

    # Tier 4
    lines.append("-" * 70)
    lines.append("TIER 4: STRUCTURAL CONSISTENCY")
    lines.append("-" * 70)
    lines.append(f"  NMI (z_inv vs spectral):   {tier4.get('nmi_mean', 0):.4f} +/- {tier4.get('nmi_std', 0):.4f}  {'PASS' if tier4.get('passes', {}).get('nmi_positive') else 'FAIL'}")
    lines.append(f"  Error-degree corr:         {tier4.get('error_degree_corr_mean', 0):.4f} +/- {tier4.get('error_degree_corr_std', 0):.4f}  (target < 0.5) {'PASS' if tier4.get('passes', {}).get('error_not_degree') else 'FAIL'}")
    bridge_ratio = tier4.get('bridge_divergence_ratio')
    bridge_str = f"{bridge_ratio:.4f}" if bridge_ratio is not None else "N/A"
    lines.append(f"  Bridge divergence ratio:   {bridge_str}  (target > 1.0) {'PASS' if tier4.get('passes', {}).get('bridge_divergence') else 'FAIL'}")
    lines.append(f"  Overall: {'PASS' if tier4.get('tier4_pass') else 'FAIL'}")
    lines.append("")

    # Baselines
    lines.append("-" * 70)
    lines.append("ABLATION BASELINES")
    lines.append("-" * 70)
    lines.append(f"  {'Baseline':<25} {'Sens':>8} {'Spec':>8} {'Prec':>8} {'Delta':>8}")
    lines.append(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for name, res in baselines.items():
        lines.append(
            f"  {name:<25} {res.get('sensitivity_mean', 0):>8.4f} "
            f"{res.get('specificity_mean', 0):>8.4f} "
            f"{res.get('precision_mean', 0):>8.4f} "
            f"{res.get('error_delta_mean', 0):>8.4f}"
        )
    # Phase 3 row
    lines.append(
        f"  {'** Phase 3 (R-GIN) **':<25} "
        f"{tier3.get('perturbation_sensitivity_mean', 0):>8.4f} "
        f"{tier3.get('perturbation_specificity_mean', 0):>8.4f} "
        f"{tier3.get('perturbation_precision_mean', 0):>8.4f} "
        f"{tier3.get('perturbation_error_delta_mean', 0):>8.4f}"
    )
    lines.append("")

    # Gate decision
    lines.append("=" * 70)
    gate_pass = gate.get("gate_pass", False)
    lines.append(f"GO/NO-GO GATE: {'GO' if gate_pass else 'NO-GO'}")
    lines.append(f"Recommendation: {gate.get('recommendation', 'Unknown')}")
    if gate.get("diagnostics"):
        lines.append("")
        lines.append("Diagnostics:")
        for d in gate["diagnostics"]:
            lines.append(f"  - {d}")
    lines.append("=" * 70)

    report_text = "\n".join(lines)

    # Save text report
    if output_dir:
        text_path = output_dir / "evaluation_report.txt"
        text_path.write_text(report_text)

    return report_text


def _strip_per_graph(d: dict) -> dict:
    """Remove per-graph arrays from report to keep JSON manageable."""
    return {k: v for k, v in d.items() if k != "recon_cosine_sim_per_graph"}


def _strip_per_repo(d: dict) -> dict:
    """Keep per_repo but ensure it's serializable."""
    result = {}
    for k, v in d.items():
        if isinstance(v, dict):
            result[k] = {str(kk): vv for kk, vv in v.items()}
        else:
            result[k] = v
    return result
