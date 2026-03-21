//! Mutation ranking dimension scorer.

use std::collections::{HashMap, HashSet};

use topo_analyzer::types::AnalysisOutput;

use crate::metrics;
use crate::signals;
use crate::types::*;

/// Score a single mutation ranking case.
pub fn score_mutation_case(
    case: &MutationCase,
    analyses: &HashMap<String, AnalysisOutput>,
) -> MutationCaseResult {
    let mutated_region: Option<HashSet<String>> = case
        .expectations
        .mutated_region
        .as_ref()
        .map(|r| r.nodes.iter().cloned().collect());

    // Evaluate each required expectation.
    let mut expectation_details = Vec::new();
    let mut passed_count = 0;

    for exp in &case.expectations.required_expectations {
        let result = eval_expectation(analyses, exp, &mutated_region);
        if result.passed {
            passed_count += 1;
        }
        expectation_details.push(result);
    }

    let total = case.expectations.required_expectations.len();
    let pairwise_accuracy = if total > 0 {
        passed_count as f64 / total as f64
    } else {
        1.0
    };

    // Repair accuracy: for each ordering pair [better, worse] where better is "repaired",
    // check if repaired is closer to clean than mutated on the primary signals.
    let has_repair = analyses.contains_key("repaired");
    let repair_accuracy = if has_repair {
        compute_repair_accuracy(case, analyses)
    } else {
        None
    };

    // Attribution@3: any of the top 3 issues in mutated overlap the mutated region.
    let attribution_at_3 = if let (Some(mutated_output), Some(region)) =
        (analyses.get("mutated"), &mutated_region)
    {
        if signals::attribution_at_k(mutated_output, region, 3) {
            1.0
        } else {
            0.0
        }
    } else {
        0.0
    };

    // Dimension score.
    let mut score_components = vec![pairwise_accuracy, attribution_at_3];
    if let Some(ra) = repair_accuracy {
        score_components.push(ra);
    }
    let score = metrics::geometric_mean(&score_components);

    MutationCaseResult {
        case_id: case.case_id.clone(),
        pairwise_accuracy,
        repair_accuracy,
        attribution_at_3,
        score,
        expectation_details,
    }
}

/// Aggregate mutation scores across all cases.
pub fn aggregate_mutation_scores(results: &[MutationCaseResult]) -> f64 {
    if results.is_empty() {
        return 0.0;
    }

    let pairwise = results.iter().map(|r| r.pairwise_accuracy).sum::<f64>() / results.len() as f64;
    let attribution = results.iter().map(|r| r.attribution_at_3).sum::<f64>() / results.len() as f64;

    let repair_results: Vec<f64> = results.iter().filter_map(|r| r.repair_accuracy).collect();
    let repair = if repair_results.is_empty() {
        1.0 // No repair cases — don't penalize.
    } else {
        repair_results.iter().sum::<f64>() / repair_results.len() as f64
    };

    metrics::geometric_mean(&[pairwise, repair, attribution])
}

/// Evaluate a single expectation.
fn eval_expectation(
    analyses: &HashMap<String, AnalysisOutput>,
    exp: &MutationExpectation,
    _mutated_region: &Option<HashSet<String>>,
) -> ExpectationResult {
    let variants = match &exp.variants {
        Some(v) if v.len() == 2 => v.clone(),
        _ => {
            return ExpectationResult {
                signal: exp.signal.clone(),
                direction: exp.direction.clone(),
                passed: false,
                left_value: None,
                right_value: None,
            };
        }
    };

    let (Some(left_output), Some(right_output)) =
        (analyses.get(&variants[0]), analyses.get(&variants[1]))
    else {
        return ExpectationResult {
            signal: exp.signal.clone(),
            direction: exp.direction.clone(),
            passed: false,
            left_value: None,
            right_value: None,
        };
    };

    let left_val = signals::extract_signal(left_output, &exp.signal, &exp.signal_args);
    let right_val = signals::extract_signal(right_output, &exp.signal, &exp.signal_args);

    let direction = exp.direction.as_deref().unwrap_or("higher_in_second");
    let passed = signals::eval_direction(&left_val, &right_val, direction, exp.margin);

    ExpectationResult {
        signal: exp.signal.clone(),
        direction: exp.direction.clone(),
        passed,
        left_value: Some(left_val.as_f64()),
        right_value: Some(right_val.as_f64()),
    }
}

/// Compute repair accuracy: fraction of repair cases where repaired is closer to
/// clean than mutated on the primary signal (cross_package_dep_count by default).
fn compute_repair_accuracy(
    case: &MutationCase,
    analyses: &HashMap<String, AnalysisOutput>,
) -> Option<f64> {
    let (Some(clean), Some(mutated), Some(repaired)) = (
        analyses.get("clean"),
        analyses.get("mutated"),
        analyses.get("repaired"),
    ) else {
        return None;
    };

    // Check repair expectations from the ordering pairs.
    let mut repair_checks = 0;
    let mut repair_passed = 0;

    for pair in &case.expectations.ordering {
        if pair.len() == 2 && pair[0] == "repaired" && pair[1] == "mutated" {
            // Repaired should be "better" than mutated on the primary signals.
            // Use cross_package_dep_count as the primary comparison.
            let rep_deps = signals::cross_package_dep_count(repaired);
            let mut_deps = signals::cross_package_dep_count(mutated);
            let clean_deps = signals::cross_package_dep_count(clean);

            repair_checks += 1;
            // Repaired is closer to clean than mutated is.
            let rep_dist = (rep_deps as f64 - clean_deps as f64).abs();
            let mut_dist = (mut_deps as f64 - clean_deps as f64).abs();
            if rep_dist <= mut_dist {
                repair_passed += 1;
            }
        }
    }

    if repair_checks == 0 {
        return None;
    }
    Some(repair_passed as f64 / repair_checks as f64)
}
