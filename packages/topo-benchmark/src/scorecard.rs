//! Scorecard assembly, loading, and saving.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};

use crate::metrics;
use crate::types::*;

/// Build a scorecard from dimension scores and guardrails.
pub fn build_scorecard(
    dimension_scores: &HashMap<String, f64>,
    guardrails: Guardrails,
    cases_passed: usize,
    cases_total: usize,
    failing_cases: Vec<String>,
) -> Scorecard {
    let scores: Vec<f64> = Dimension::all()
        .iter()
        .filter_map(|d| dimension_scores.get(d.as_str()).copied())
        .collect();
    let overall_primary = metrics::geometric_mean(&scores);

    let promotion_decision = if guardrails.all_pass() && failing_cases.is_empty() {
        "pass".to_string()
    } else {
        "fail".to_string()
    };

    Scorecard {
        runner_version: RUNNER_VERSION.to_string(),
        overall_primary,
        dimensions: dimension_scores.clone(),
        guardrails,
        cases_passed,
        cases_total,
        failing_cases,
        promotion_decision,
    }
}

/// Load a scorecard from disk.
pub fn load_scorecard(path: &Path) -> Result<Scorecard> {
    let text =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))
}

/// Save a scorecard to disk.
pub fn save_scorecard(scorecard: &Scorecard, path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let json = serde_json::to_string_pretty(scorecard)?;
    std::fs::write(path, json).with_context(|| format!("writing {}", path.display()))
}
