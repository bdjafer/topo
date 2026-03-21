//! Architecture recovery dimension scorer.

use std::collections::{HashMap, HashSet};

use topo_analyzer::types::{AnalysisOutput, AnalyzerInput};

use crate::metrics;
use crate::signals;
use crate::types::*;

/// Score a single architecture recovery case.
pub fn score_architecture_case(
    input: &AnalyzerInput,
    labels: &ArchitectureLabels,
    output: &AnalysisOutput,
    baseline_partitions: &HashMap<String, Partition>,
) -> ArchitectureCaseResult {
    let case_id = String::new(); // Set by caller.

    // Gold partition: node_id → label string → usize.
    let mut gold_label_to_id: HashMap<&str, usize> = HashMap::new();
    let mut next_id = 0;
    let mut gold_partition: HashMap<String, usize> = HashMap::new();
    for (node_id, label) in &labels.included_nodes {
        let id = *gold_label_to_id.entry(label.as_str()).or_insert_with(|| {
            let id = next_id;
            next_id += 1;
            id
        });
        gold_partition.insert(node_id.clone(), id);
    }

    let gold_nodes: HashSet<String> = labels.included_nodes.keys().cloned().collect();

    // Predicted partition (filtered to included nodes).
    let full_pred = signals::partition_labels(output);
    let pred: HashMap<String, usize> = full_pred
        .into_iter()
        .filter(|(k, _)| gold_nodes.contains(k))
        .collect();

    // NMI.
    let nmi = metrics::compute_nmi(&pred, &gold_partition);

    // Boundary F1 — edges between included nodes.
    let edges: Vec<(String, String)> = input
        .edges
        .iter()
        .filter(|e| gold_nodes.contains(&e.source) && gold_nodes.contains(&e.target))
        .map(|e| (e.source.clone(), e.target.clone()))
        .collect();
    let boundary_f1 = metrics::compute_boundary_f1(&edges, &pred, &gold_partition);

    // Coverage.
    let coverage = metrics::compute_coverage(&pred, &gold_nodes);

    // Baseline NMI.
    let mut baseline_nmi = HashMap::new();
    for (name, base_part) in baseline_partitions {
        let base_filtered: HashMap<String, usize> = base_part
            .iter()
            .filter(|(k, _)| gold_nodes.contains(*k))
            .map(|(k, v)| (k.clone(), *v))
            .collect();
        baseline_nmi.insert(name.clone(), metrics::compute_nmi(&base_filtered, &gold_partition));
    }

    // Guardrails.
    let coverage_ok = coverage >= COVERAGE_FLOOR;
    let baseline_ok = baseline_nmi
        .get("directory")
        .map_or(true, |&dir_nmi| nmi >= dir_nmi);

    // Dimension score.
    let score = metrics::geometric_mean(&[nmi, boundary_f1])
        * (coverage / COVERAGE_FLOOR).min(1.0);

    ArchitectureCaseResult {
        case_id,
        nmi,
        boundary_f1,
        coverage,
        score,
        guardrails: ArchGuardrails { coverage_ok, baseline_ok },
        baseline_nmi,
    }
}

/// Compute the architecture recovery dimension score from case results.
pub fn aggregate_architecture_scores(results: &[ArchitectureCaseResult]) -> (f64, ArchGuardrails) {
    if results.is_empty() {
        return (0.0, ArchGuardrails { coverage_ok: true, baseline_ok: true });
    }
    let avg_score = results.iter().map(|r| r.score).sum::<f64>() / results.len() as f64;
    let coverage_ok = results.iter().all(|r| r.guardrails.coverage_ok);
    let baseline_ok = results.iter().all(|r| r.guardrails.baseline_ok);
    (avg_score, ArchGuardrails { coverage_ok, baseline_ok })
}
