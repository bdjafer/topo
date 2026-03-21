//! Markdown report generation.

use std::collections::HashMap;

use crate::types::*;

/// Generate a markdown summary report.
pub fn generate_summary(
    scorecard: &Scorecard,
    _dimension_details: &HashMap<String, serde_json::Value>,
    _baseline_results: &HashMap<String, serde_json::Value>,
    comparison: Option<&ComparisonResult>,
) -> String {
    let mut out = String::new();

    out.push_str("# Benchmark Report\n\n");
    out.push_str(&format!("Runner version: {}\n\n", scorecard.runner_version));

    // Overall.
    out.push_str("## Overall\n\n");
    out.push_str(&format!(
        "| Metric | Value |\n|---|---|\n| Overall Primary | {:.4} |\n| Decision | {} |\n| Cases | {}/{} passed |\n\n",
        scorecard.overall_primary,
        scorecard.promotion_decision,
        scorecard.cases_passed,
        scorecard.cases_total,
    ));

    // Dimensions.
    out.push_str("## Dimensions\n\n");
    out.push_str("| Dimension | Score |\n|---|---|\n");
    for dim in Dimension::all() {
        if let Some(score) = scorecard.dimensions.get(dim.as_str()) {
            out.push_str(&format!("| {} | {:.4} |\n", dim.as_str(), score));
        }
    }
    out.push('\n');

    // Guardrails.
    out.push_str("## Guardrails\n\n");
    let g = &scorecard.guardrails;
    out.push_str(&format!("| Check | Status |\n|---|---|\n"));
    out.push_str(&format!(
        "| Coverage | {} |\n| Baseline | {} |\n| No Regressions | {} |\n| False Positive | {} |\n| No Anomaly Flood | {} |\n\n",
        pass_fail(g.coverage_ok),
        pass_fail(g.baseline_ok),
        pass_fail(g.no_regressions),
        pass_fail(g.false_positive_ok),
        pass_fail(g.no_anomaly_flood),
    ));

    // Failing cases.
    if !scorecard.failing_cases.is_empty() {
        out.push_str("## Failing Cases\n\n");
        for case in &scorecard.failing_cases {
            out.push_str(&format!("- {case}\n"));
        }
        out.push('\n');
    }

    // Comparison.
    if let Some(comp) = comparison {
        out.push_str("## Comparison\n\n");
        out.push_str(&format!(
            "Overall: {:.4} → {:.4} (Δ {:.4})\n\n",
            comp.reference_overall, comp.candidate_overall, comp.overall_delta
        ));
        out.push_str("| Dimension | Reference | Candidate | Delta | Status |\n|---|---|---|---|---|\n");
        for dim in Dimension::all() {
            if let Some(d) = comp.dimensions.get(dim.as_str()) {
                let status = if d.regressed { "REGRESSION" } else { "OK" };
                out.push_str(&format!(
                    "| {} | {:.4} | {:.4} | {:.4} | {} |\n",
                    dim.as_str(), d.reference, d.candidate, d.delta, status
                ));
            }
        }
        out.push_str(&format!("\nDecision: {}\n", comp.promotion_decision));
    }

    out
}

fn pass_fail(ok: bool) -> &'static str {
    if ok { "PASS" } else { "FAIL" }
}
