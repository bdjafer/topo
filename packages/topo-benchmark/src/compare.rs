//! Compare two benchmark runs.

use std::collections::HashMap;
use std::path::Path;

use anyhow::Result;

use crate::scorecard;
use crate::types::*;

/// Compare candidate vs reference benchmark runs.
pub fn compare_runs(
    candidate_dir: &Path,
    reference_dir: &Path,
    max_regression: f64,
) -> Result<ComparisonResult> {
    let candidate = scorecard::load_scorecard(&candidate_dir.join("scorecard.json"))?;
    let reference = scorecard::load_scorecard(&reference_dir.join("scorecard.json"))?;

    let overall_delta = candidate.overall_primary - reference.overall_primary;

    let mut dimensions = HashMap::new();
    let mut has_regression = false;

    for dim in Dimension::all() {
        let key = dim.as_str();
        let cand_score = candidate.dimensions.get(key).copied().unwrap_or(0.0);
        let ref_score = reference.dimensions.get(key).copied().unwrap_or(0.0);
        let delta = cand_score - ref_score;
        let regressed = delta < -max_regression;
        if regressed {
            has_regression = true;
        }
        dimensions.insert(
            key.to_string(),
            DimensionDelta {
                candidate: cand_score,
                reference: ref_score,
                delta,
                regressed,
            },
        );
    }

    let overall_improved = overall_delta >= 0.0;
    let guardrails_pass = candidate.guardrails.all_pass();

    // Check that no previously-passing cases now fail.
    let no_case_regressions = check_no_case_regressions(&candidate, &reference);

    let promotion_decision = if overall_improved
        && !has_regression
        && guardrails_pass
        && no_case_regressions
    {
        "pass".to_string()
    } else {
        "fail".to_string()
    };

    Ok(ComparisonResult {
        overall_delta,
        candidate_overall: candidate.overall_primary,
        reference_overall: reference.overall_primary,
        dimensions,
        promotion_decision,
        reasons: ComparisonReasons {
            overall_improved,
            no_regressions: !has_regression,
            guardrails_pass,
        },
    })
}

/// Check that no previously-passing cases now fail.
fn check_no_case_regressions(candidate: &Scorecard, reference: &Scorecard) -> bool {
    // If the reference had no failing cases but the candidate does, that's a regression.
    // More precisely: every case that was NOT in reference.failing_cases should also
    // NOT be in candidate.failing_cases.
    for failing in &candidate.failing_cases {
        if !reference.failing_cases.contains(failing) {
            return false;
        }
    }
    true
}
