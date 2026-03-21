//! Stability dimension scorer.

use std::collections::HashMap;

use topo_analyzer::types::AnalysisOutput;

use crate::metrics;
use crate::signals;
use crate::types::*;

/// Score a single stability case (base + all perturbations).
pub fn score_stability_case(
    case: &StabilityCase,
    base_output: &AnalysisOutput,
    pert_outputs: &HashMap<String, AnalysisOutput>,
) -> StabilityCaseResult {
    let base_partition = signals::partition_labels(base_output);
    let base_roles = signals::role_map(base_output);

    let mut per_perturbation = HashMap::new();
    let mut partition_aris = Vec::new();
    let mut role_f1s = Vec::new();

    for (pert_name, pert_output) in pert_outputs {
        let pert_partition = signals::partition_labels(pert_output);
        let pert_roles = signals::role_map(pert_output);

        // If there's a node mapping, remap the perturbation partition.
        let (mapped_pert_partition, mapped_pert_roles) =
            if let Some(mapping) = case.node_mappings.get(pert_name) {
                remap_partition(&pert_partition, &pert_roles, mapping)
            } else {
                (pert_partition, pert_roles)
            };

        let partition_ari = metrics::compute_ari(&base_partition, &mapped_pert_partition);
        let role_f1 = metrics::compute_role_macro_f1(&base_roles, &mapped_pert_roles);

        partition_aris.push(partition_ari);
        role_f1s.push(role_f1);

        per_perturbation.insert(
            pert_name.clone(),
            PerturbationResult { partition_ari, role_f1 },
        );
    }

    let partition_stability = if partition_aris.is_empty() {
        1.0
    } else {
        partition_aris.iter().sum::<f64>() / partition_aris.len() as f64
    };

    let role_stability = if role_f1s.is_empty() {
        1.0
    } else {
        role_f1s.iter().sum::<f64>() / role_f1s.len() as f64
    };

    let score = metrics::geometric_mean(&[partition_stability, role_stability]);

    StabilityCaseResult {
        case_id: case.case_id.clone(),
        partition_stability,
        role_stability,
        score,
        per_perturbation,
    }
}

/// Aggregate stability scores across all cases.
pub fn aggregate_stability_scores(results: &[StabilityCaseResult]) -> f64 {
    if results.is_empty() {
        return 0.0;
    }
    let avg_partition = results.iter().map(|r| r.partition_stability).sum::<f64>()
        / results.len() as f64;
    let avg_role = results.iter().map(|r| r.role_stability).sum::<f64>()
        / results.len() as f64;
    metrics::geometric_mean(&[avg_partition, avg_role])
}

/// Remap perturbation results using node mapping (base_id → pert_id inverted).
fn remap_partition(
    pert_partition: &Partition,
    pert_roles: &HashMap<String, String>,
    mapping: &HashMap<String, String>,
) -> (Partition, HashMap<String, String>) {
    // mapping is base_id → pert_id. We need pert_id → base_id to remap.
    let reverse: HashMap<&str, &str> = mapping
        .iter()
        .map(|(base, pert)| (pert.as_str(), base.as_str()))
        .collect();

    let mut mapped_partition = Partition::new();
    for (pert_id, &cluster) in pert_partition {
        let base_id = reverse
            .get(pert_id.as_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| pert_id.clone());
        mapped_partition.insert(base_id, cluster);
    }

    let mut mapped_roles = HashMap::new();
    for (pert_id, role) in pert_roles {
        let base_id = reverse
            .get(pert_id.as_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| pert_id.clone());
        mapped_roles.insert(base_id, role.clone());
    }

    (mapped_partition, mapped_roles)
}
