//! Seeded anomaly detection dimension scorer.

use std::collections::HashSet;

use topo_analyzer::types::AnalysisOutput;

use crate::metrics;
use crate::types::*;

/// Score a single anomaly case (with gold anomalies).
pub fn score_anomaly_case(
    case: &AnomalyCase,
    output: &AnalysisOutput,
) -> AnomalyCaseResult {
    let gold = match &case.gold {
        Some(g) => g,
        None => {
            // Clean graph — no anomalies expected.
            let is_clean = score_clean_graph(output);
            return AnomalyCaseResult {
                case_id: case.case_id.clone(),
                average_precision: if is_clean { 1.0 } else { 0.0 },
                precision_at_3: if is_clean { 1.0 } else { 0.0 },
                score: if is_clean { 1.0 } else { 0.0 },
                n_predicted: output.issues.len(),
                n_gold: 0,
                is_clean_graph: true,
            };
        }
    };

    let n_gold = gold.anomalies.len();
    let n_predicted = output.issues.len();

    if n_gold == 0 {
        return AnomalyCaseResult {
            case_id: case.case_id.clone(),
            average_precision: 1.0,
            precision_at_3: 1.0,
            score: 1.0,
            n_predicted,
            n_gold: 0,
            is_clean_graph: false,
        };
    }

    // Sort predicted issues by severity descending.
    let mut sorted_issues: Vec<&topo_analyzer::types::IssueOutput> =
        output.issues.iter().collect();
    sorted_issues.sort_by(|a, b| {
        b.severity
            .partial_cmp(&a.severity)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Match each predicted issue to gold anomalies.
    let mut items: Vec<(f64, bool)> = Vec::new();
    let mut matched_gold: HashSet<usize> = HashSet::new();

    for issue in &sorted_issues {
        let issue_nodes: HashSet<String> =
            issue.anchors.iter().map(|a| a.node_id.clone()).collect();

        let mut is_match = false;
        for (gi, gold_anomaly) in gold.anomalies.iter().enumerate() {
            if matched_gold.contains(&gi) {
                continue;
            }
            let gold_nodes = gold_anomaly.node_set();

            // Check kind match (when specified).
            let kind_ok = gold_anomaly
                .kind
                .as_ref()
                .map_or(true, |k| issue.kind == *k);

            if kind_ok && metrics::node_set_iou_hierarchical(&issue_nodes, &gold_nodes) >= IOU_THRESHOLD {
                is_match = true;
                matched_gold.insert(gi);
                break;
            }
        }
        items.push((issue.severity, is_match));
    }

    let average_precision = metrics::compute_average_precision(&items);
    let precision_at_3 = metrics::compute_precision_at_k(&items, 3);
    let score = metrics::geometric_mean(&[average_precision, precision_at_3]);

    AnomalyCaseResult {
        case_id: case.case_id.clone(),
        average_precision,
        precision_at_3,
        score,
        n_predicted,
        n_gold,
        is_clean_graph: false,
    }
}

/// Score a clean (false-positive test) graph. Returns true if pass.
pub fn score_clean_graph(output: &AnalysisOutput) -> bool {
    !output
        .issues
        .iter()
        .any(|i| i.severity > HIGH_SEVERITY_THRESHOLD)
}

/// Check anomaly flood guardrail. Returns true if pass.
pub fn check_anomaly_flood(candidate_count: usize, reference_count: usize) -> bool {
    if reference_count == 0 {
        return candidate_count == 0;
    }
    (candidate_count as f64) <= (reference_count as f64) * ANOMALY_FLOOD_RATIO
}

/// Aggregate anomaly scores across all cases.
pub fn aggregate_anomaly_scores(results: &[AnomalyCaseResult]) -> f64 {
    if results.is_empty() {
        return 0.0;
    }
    let avg_ap = results.iter().map(|r| r.average_precision).sum::<f64>() / results.len() as f64;
    let avg_p3 = results.iter().map(|r| r.precision_at_3).sum::<f64>() / results.len() as f64;
    metrics::geometric_mean(&[avg_ap, avg_p3])
}
