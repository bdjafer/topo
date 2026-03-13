"""Generate human-readable benchmark reports."""

from __future__ import annotations


def generate_summary(
    scorecard: dict,
    dimension_details: dict[str, dict] | None = None,
    baseline_results: dict[str, dict] | None = None,
    comparison: dict | None = None,
) -> str:
    """Generate a markdown summary report."""
    lines: list[str] = []
    lines.append("# Benchmark Report")
    lines.append("")
    lines.append(f"**Tier:** {scorecard['tier']}")
    lines.append(f"**Split:** {scorecard['split']}")
    lines.append(f"**Overall Primary Score:** {scorecard['overall_primary']:.4f}")
    lines.append(f"**Promotion Decision:** {scorecard['promotion_decision']}")
    lines.append("")

    # Dimensions table
    lines.append("## Dimension Scores")
    lines.append("")
    lines.append("| Dimension | Score |")
    lines.append("|-----------|-------|")
    for dim, score in scorecard.get("dimensions", {}).items():
        lines.append(f"| {dim} | {score:.4f} |")
    lines.append("")

    # Guardrails
    lines.append("## Guardrails")
    lines.append("")
    for name, passed in scorecard.get("guardrails", {}).items():
        status = "PASS" if passed else "FAIL"
        lines.append(f"- {name}: **{status}**")
    lines.append("")

    # Dimension details
    if dimension_details:
        lines.append("## Dimension Details")
        lines.append("")
        for dim, details in dimension_details.items():
            lines.append(f"### {dim}")
            lines.append("")
            for key, value in details.items():
                if key in ("per_case", "guardrails", "baseline_scores"):
                    continue
                if isinstance(value, float):
                    lines.append(f"- {key}: {value:.4f}")
                else:
                    lines.append(f"- {key}: {value}")
            lines.append("")

    # Baseline comparison
    if baseline_results:
        lines.append("## Baseline Comparison")
        lines.append("")
        for name, results in baseline_results.items():
            lines.append(f"### {name}")
            for key, value in results.items():
                if isinstance(value, float):
                    lines.append(f"- {key}: {value:.4f}")
                else:
                    lines.append(f"- {key}: {value}")
            lines.append("")

    # Comparison (if available)
    if comparison:
        lines.append("## Comparison")
        lines.append("")
        lines.append(f"**Overall Delta:** {comparison['overall_delta']:+.4f}")
        lines.append(f"**Promotion:** {comparison['promotion_decision']}")
        lines.append("")
        lines.append("| Dimension | Candidate | Reference | Delta | CI |")
        lines.append("|-----------|-----------|-----------|-------|----|")
        for dim, d in comparison.get("dimensions", {}).items():
            lines.append(
                f"| {dim} | {d['candidate']:.4f} | {d['reference']:.4f} | "
                f"{d['delta']:+.4f} | [{d['ci_low']:+.4f}, {d['ci_high']:+.4f}] |"
            )
        lines.append("")

    return "\n".join(lines)
