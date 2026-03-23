"""Go/No-Go gate for Phase 3 deployment.

Aggregates results from all evaluation tiers and baselines to make a
binary deployment decision.
"""


def go_no_go_gate(
    tier1: dict,
    tier2: dict,
    tier3: dict,
    tier4: dict,
    baselines: dict,
) -> dict:
    """Apply the go/no-go gate from PHASE_3.md.

    Gate checks:
    1. Tier 1 intrinsic metrics pass.
    2. Tier 2 agreement: rank correlation > 0.3.
    3. Tier 3 sensitivity > 0.6, specificity > 0.75.
    4. Tier 3 precision > Phase 2 precision + 0.10 (PHASE_3.md requires precision improvement).
    5. Phase 3 beats all ablation baselines on precision.
    6. Control perturbation sensitivity ~ 0.20 ± 0.05.

    Args:
        tier1: Results from tier1_intrinsic_metrics.
        tier2: Results from tier2_phase2_agreement.
        tier3: Results from tier3_perturbation_test.
        tier4: Results from tier4_structural_consistency.
        baselines: Results from run_baselines.

    Returns:
        Dict with gate decision, per-check results, and diagnostic info.
    """
    checks = {}

    # Check 1: Tier 1 intrinsic metrics
    checks["tier1_pass"] = tier1.get("tier1_pass", False)

    # Check 2: Tier 2 agreement
    checks["tier2_agreement"] = tier2.get("tier2_pass", False)

    # Check 3: Tier 3 sensitivity and specificity
    checks["tier3_sensitivity"] = (
        tier3.get("perturbation_sensitivity_mean", 0) > 0.6
    )
    checks["tier3_specificity"] = (
        tier3.get("perturbation_specificity_mean", 0) > 0.75
    )

    # Check 4: Precision improvement over Phase 2
    p3_precision = tier3.get("perturbation_precision_mean", 0)
    p2_precision = baselines.get("phase2_local_variation", {}).get("precision_mean", 0)
    precision_delta = p3_precision - p2_precision
    checks["precision_improvement"] = precision_delta >= 0.10
    checks["precision_delta"] = float(precision_delta)

    # Check 5: Phase 3 beats all baselines on precision
    beats_all = True
    baseline_comparisons = {}
    for name, baseline_result in baselines.items():
        baseline_prec = baseline_result.get("precision_mean", 0)
        beats = p3_precision > baseline_prec
        baseline_comparisons[name] = {
            "baseline_precision": float(baseline_prec),
            "phase3_precision": float(p3_precision),
            "phase3_wins": beats,
        }
        if not beats:
            beats_all = False
    checks["beats_all_baselines"] = beats_all
    checks["baseline_comparisons"] = baseline_comparisons

    # Check 6: Control perturbation sanity
    ctrl_sens = tier3.get("control_sensitivity_mean", 0)
    checks["control_sanity"] = 0.15 <= ctrl_sens <= 0.25

    # Overall gate
    gate_pass = all([
        checks["tier1_pass"],
        checks["tier2_agreement"],
        checks["tier3_sensitivity"],
        checks["tier3_specificity"],
        checks["precision_improvement"],
        checks["beats_all_baselines"],
        checks["control_sanity"],
    ])

    # Diagnostics for failures
    diagnostics = []
    if not checks["tier1_pass"]:
        diagnostics.append("Tier 1 intrinsic metrics failed — model may need more training.")
    if not checks["tier2_agreement"]:
        diagnostics.append("Tier 2 Phase 2 agreement failed — model may have learned spurious patterns.")
    if not checks["tier3_sensitivity"]:
        diagnostics.append("Tier 3 sensitivity failed — model lacks discrimination for perturbations.")
    if not checks["tier3_specificity"]:
        diagnostics.append("Tier 3 specificity failed — model over-flags non-swapped nodes.")
    if not checks["precision_improvement"]:
        diagnostics.append(
            f"Precision improvement failed: Phase 3 precision={p3_precision:.3f}, "
            f"Phase 2 precision={p2_precision:.3f}, delta={precision_delta:.3f} (need >= 0.10)."
        )
    if not checks["beats_all_baselines"]:
        losers = [n for n, c in baseline_comparisons.items() if not c["phase3_wins"]]
        diagnostics.append(f"Phase 3 didn't beat baselines: {losers}")
    if not checks["control_sanity"]:
        diagnostics.append(
            f"Control perturbation sensitivity={ctrl_sens:.3f} outside expected range [0.15, 0.25]."
        )

    return {
        "gate_pass": gate_pass,
        "checks": checks,
        "diagnostics": diagnostics,
        "recommendation": "DEPLOY Phase 3" if gate_pass else "DO NOT DEPLOY — return to Step 2",
    }
