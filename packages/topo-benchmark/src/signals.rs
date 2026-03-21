//! Signal extraction from AnalysisOutput.

use std::collections::{HashMap, HashSet};

use topo_analyzer::types::AnalysisOutput;

use crate::metrics;
use crate::types::{SignalValue, IOU_THRESHOLD, Partition};

/// Extract module partition labels (non-unassigned modules only).
pub fn partition_labels(output: &AnalysisOutput) -> Partition {
    let mut partition = Partition::new();
    for module in &output.architecture.modules {
        if module.unassigned {
            continue;
        }
        for member in &module.members {
            partition.insert(member.clone(), module.id);
        }
    }
    partition
}

/// ARI between module partitions of two analyses on shared nodes.
pub fn partition_similarity(a: &AnalysisOutput, b: &AnalysisOutput) -> f64 {
    let pa = partition_labels(a);
    let pb = partition_labels(b);
    metrics::compute_ari(&pa, &pb)
}

/// Largest module ratio: max(module.size) / sum(module.size).
pub fn largest_module_ratio(output: &AnalysisOutput) -> f64 {
    let sizes: Vec<usize> = output
        .architecture
        .modules
        .iter()
        .filter(|m| !m.unassigned)
        .map(|m| m.size)
        .collect();
    let total: usize = sizes.iter().sum();
    if total == 0 {
        return 0.0;
    }
    *sizes.iter().max().unwrap() as f64 / total as f64
}

/// Number of non-unassigned modules.
pub fn module_count(output: &AnalysisOutput) -> usize {
    output
        .architecture
        .modules
        .iter()
        .filter(|m| !m.unassigned)
        .count()
}

/// Max severity among issues matching the given kind, default 0.
pub fn max_issue_severity(output: &AnalysisOutput, kind: &str) -> f64 {
    output
        .issues
        .iter()
        .filter(|i| i.kind == kind)
        .map(|i| i.severity)
        .fold(0.0f64, f64::max)
}

/// Predicted structural role for a specific node.
pub fn target_role(output: &AnalysisOutput, node_id: &str) -> Option<String> {
    output
        .roles
        .iter()
        .find(|r| r.node_id == node_id)
        .map(|r| r.role.clone())
}

/// Whether any spectral_outlier issue contains the given node.
pub fn target_has_spectral_outlier(output: &AnalysisOutput, node_id: &str) -> bool {
    output.issues.iter().any(|i| {
        i.kind == "spectral_outlier" && i.anchors.iter().any(|a| a.node_id == node_id)
    })
}

/// Whether any of the top-k issues (by severity) overlap the mutated region.
pub fn attribution_at_k(
    output: &AnalysisOutput,
    mutated_region: &HashSet<String>,
    k: usize,
) -> bool {
    if mutated_region.is_empty() {
        return false;
    }
    let mut issues: Vec<&topo_analyzer::types::IssueOutput> = output.issues.iter().collect();
    issues.sort_by(|a, b| b.severity.partial_cmp(&a.severity).unwrap_or(std::cmp::Ordering::Equal));

    for issue in issues.iter().take(k) {
        let issue_nodes: HashSet<String> =
            issue.anchors.iter().map(|a| a.node_id.clone()).collect();
        if metrics::node_set_iou(&issue_nodes, mutated_region) >= IOU_THRESHOLD {
            return true;
        }
    }
    false
}

/// Count of cross-package dependencies.
pub fn cross_package_dep_count(output: &AnalysisOutput) -> usize {
    output.architecture.dependencies.len()
}

/// Whether a specific cross-package dependency exists (by module label prefix).
pub fn has_cross_package_dep(
    output: &AnalysisOutput,
    source_pkg: &str,
    target_pkg: &str,
) -> bool {
    let modules = &output.architecture.modules;
    for dep in &output.architecture.dependencies {
        let src_mod = modules.iter().find(|m| m.id == dep.source);
        let tgt_mod = modules.iter().find(|m| m.id == dep.target);
        if let (Some(src), Some(tgt)) = (src_mod, tgt_mod) {
            let src_label = &src.label;
            let tgt_label = &tgt.label;
            if (src_label.starts_with(source_pkg) || src_label == source_pkg)
                && (tgt_label.starts_with(target_pkg) || tgt_label == target_pkg)
            {
                return true;
            }
        }
    }
    false
}

/// Whether the analysis produced an issue of the given kind.
pub fn has_finding(output: &AnalysisOutput, kind: &str) -> bool {
    output.issues.iter().any(|i| i.kind == kind)
}

/// Count of issues matching the given kind.
pub fn finding_count(output: &AnalysisOutput, kind: &str) -> usize {
    output.issues.iter().filter(|i| i.kind == kind).count()
}

/// Extract roles as a map from node_id to role string.
pub fn role_map(output: &AnalysisOutput) -> HashMap<String, String> {
    output
        .roles
        .iter()
        .map(|r| (r.node_id.clone(), r.role.clone()))
        .collect()
}

// ── Signal dispatch ──────────────────────────────────────────────────────────

/// Extract a named signal from analysis output.
pub fn extract_signal(
    output: &AnalysisOutput,
    name: &str,
    args: &Option<HashMap<String, serde_json::Value>>,
) -> SignalValue {
    match name {
        "cross_package_dep_count" => SignalValue::Count(cross_package_dep_count(output)),
        "has_cross_package_dep" => {
            let (src, tgt) = extract_pkg_args(args);
            SignalValue::Bool(has_cross_package_dep(output, &src, &tgt))
        }
        "has_finding" => {
            let kind = extract_kind_arg(args);
            SignalValue::Bool(has_finding(output, &kind))
        }
        "finding_count" => {
            let kind = extract_kind_arg(args);
            SignalValue::Count(finding_count(output, &kind))
        }
        "largest_module_ratio" => SignalValue::Float(largest_module_ratio(output)),
        "module_count" => SignalValue::Count(module_count(output)),
        "max_cross_module_severity" => {
            SignalValue::Float(max_issue_severity(output, "cross_module"))
        }
        "max_cycle_severity" => {
            SignalValue::Float(max_issue_severity(output, "cycle_member"))
        }
        "target_role" => {
            let node = extract_string_arg(args, "node_id");
            SignalValue::Role(target_role(output, &node))
        }
        "target_has_spectral_outlier" => {
            let node = extract_string_arg(args, "node_id");
            SignalValue::Bool(target_has_spectral_outlier(output, &node))
        }
        "attribution_at_3" => {
            // Mutated region must be passed separately; this returns false as default.
            // The mutation scorer handles this directly.
            SignalValue::Bool(false)
        }
        "partition_similarity_to_clean" => {
            // Handled by the mutation scorer directly (needs two outputs).
            SignalValue::Float(0.0)
        }
        _ => {
            eprintln!("warning: unknown signal '{name}', defaulting to 0.0");
            SignalValue::Float(0.0)
        }
    }
}

/// Evaluate a direction constraint between two signal values.
pub fn eval_direction(
    left: &SignalValue,
    right: &SignalValue,
    direction: &str,
    margin: f64,
) -> bool {
    match direction {
        "higher_in_second" => right.as_f64() > left.as_f64() + margin,
        "lower_in_second" => left.as_f64() > right.as_f64() + margin,
        "true_in_second" => right.as_bool(),
        "false_in_second" => !right.as_bool(),
        "present_in_second_not_first" => right.as_bool() && !left.as_bool(),
        "equal" => (right.as_f64() - left.as_f64()).abs() <= margin,
        _ => {
            eprintln!("warning: unknown direction '{direction}'");
            false
        }
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────────

fn extract_pkg_args(args: &Option<HashMap<String, serde_json::Value>>) -> (String, String) {
    let args = args.as_ref();
    let src = args
        .and_then(|a| a.get("source_pkg"))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let tgt = args
        .and_then(|a| a.get("target_pkg"))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    (src, tgt)
}

fn extract_kind_arg(args: &Option<HashMap<String, serde_json::Value>>) -> String {
    args.as_ref()
        .and_then(|a| a.get("kind"))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string()
}

fn extract_string_arg(args: &Option<HashMap<String, serde_json::Value>>, key: &str) -> String {
    args.as_ref()
        .and_then(|a| a.get(key))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string()
}
