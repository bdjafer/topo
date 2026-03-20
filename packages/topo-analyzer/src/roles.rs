//! Structural role classification.
//!
//! Classifies code entities by their structural position in the graph:
//! hub, bridge, utility, entry point, orphan, regular.
//!
//! Uses distribution-based classification: percentile ranks of degree
//! and betweenness, plus directional flow score.
//!
//! Ports logic from Python topo_analyzer.roles.

use crate::graph::Graph;
use crate::types::RoleOutput;

/// Percentile above which a metric is considered "high".
const PCT_THRESHOLD: f64 = 0.9;
/// Minimum edges for directional roles.
const MIN_DIRECTIONAL_DEGREE: usize = 3;
/// Flow imbalance threshold for directional roles.
const DIRECTION_THRESHOLD: f64 = 0.6;
/// Hub must exceed median degree by this gap.
const MIN_HUB_GAP: f64 = 2.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    Hub,
    Bridge,
    Utility,
    EntryPoint,
    Orphan,
    Regular,
}

impl Role {
    pub fn as_str(&self) -> &'static str {
        match self {
            Role::Hub => "hub",
            Role::Bridge => "bridge",
            Role::Utility => "utility",
            Role::EntryPoint => "entry_point",
            Role::Orphan => "orphan",
            Role::Regular => "regular",
        }
    }
}

/// Classify structural roles for all nodes.
pub fn classify_roles(graph: &Graph, betweenness: &[f64]) -> Vec<RoleOutput> {
    let n = graph.n;
    if n == 0 {
        return Vec::new();
    }

    let mut degrees = vec![0usize; n];
    let mut in_degrees = vec![0usize; n];
    let mut out_degrees = vec![0usize; n];

    for i in 0..n {
        in_degrees[i] = graph.in_degree(i);
        out_degrees[i] = graph.out_degree(i);
        degrees[i] = in_degrees[i] + out_degrees[i];
    }

    let degree_f: Vec<f64> = degrees.iter().map(|&d| d as f64).collect();
    let degree_pcts = percentile_ranks(&degree_f);
    let btw_pcts = percentile_ranks(betweenness);
    let median_degree = median(&degree_f);

    let mut roles = Vec::with_capacity(n);
    for i in 0..n {
        let role = classify_node(
            degrees[i],
            in_degrees[i],
            out_degrees[i],
            degree_pcts[i],
            btw_pcts[i],
            median_degree,
        );
        roles.push(RoleOutput {
            node_id: graph.node_ids[i].clone(),
            role: role.as_str().to_string(),
            degree: degrees[i],
            betweenness: round4(betweenness[i]),
            in_degree: in_degrees[i],
            out_degree: out_degrees[i],
            anchor: Some(graph.anchor(i)),
        });
    }
    roles
}

/// Classify a single node using distribution-based rules.
///
/// Priority:
/// 1. ORPHAN: degree == 0
/// 2. BRIDGE: high betweenness, not high degree
/// 3. UTILITY: strong sink (direction ≤ -0.6)
/// 4. ENTRY_POINT: strong source (direction ≥ +0.6)
/// 5. HUB: high degree, above median gap, balanced flow
/// 6. REGULAR: everything else
fn classify_node(
    degree: usize,
    in_degree: usize,
    out_degree: usize,
    degree_pct: f64,
    betweenness_pct: f64,
    median_degree: f64,
) -> Role {
    if degree == 0 {
        return Role::Orphan;
    }

    let direction = (out_degree as f64 - in_degree as f64) / degree as f64;

    // Bridge: high betweenness but not high degree.
    if betweenness_pct >= PCT_THRESHOLD && degree_pct < PCT_THRESHOLD {
        return Role::Bridge;
    }

    // Directional roles: strong flow imbalance with enough edges.
    if degree >= MIN_DIRECTIONAL_DEGREE && direction <= -DIRECTION_THRESHOLD {
        return Role::Utility;
    }
    if degree >= MIN_DIRECTIONAL_DEGREE && direction >= DIRECTION_THRESHOLD {
        return Role::EntryPoint;
    }

    // Hub: top-percentile degree, clearly above median, balanced flow.
    if degree_pct >= PCT_THRESHOLD
        && degree as f64 >= median_degree + MIN_HUB_GAP
        && direction.abs() < DIRECTION_THRESHOLD
    {
        return Role::Hub;
    }

    Role::Regular
}

/// Compute percentile ranks: fraction of values strictly less.
///
/// Uses ranks/(n-1) so maximum maps to 1.0, minimum to 0.0.
/// Matches Python `roles.py::_percentile_ranks`.
pub fn percentile_ranks(values: &[f64]) -> Vec<f64> {
    let n = values.len();
    if n <= 1 {
        return vec![0.0; n];
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    values
        .iter()
        .map(|&v| {
            // Count strictly less via binary search (left side).
            let rank = sorted.partition_point(|&x| x < v);
            rank as f64 / (n - 1) as f64
        })
        .collect()
}

fn median(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 0 {
        (sorted[mid - 1] + sorted[mid]) / 2.0
    } else {
        sorted[mid]
    }
}

fn round4(v: f64) -> f64 {
    (v * 10000.0).round() / 10000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_percentile_ranks_basic() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let pcts = percentile_ranks(&values);
        assert_eq!(pcts[0], 0.0); // min
        assert_eq!(pcts[4], 1.0); // max
    }

    #[test]
    fn test_percentile_ranks_ties() {
        let values = vec![1.0, 1.0, 1.0];
        let pcts = percentile_ranks(&values);
        assert_eq!(pcts[0], 0.0);
        assert_eq!(pcts[1], 0.0);
        assert_eq!(pcts[2], 0.0);
    }

    #[test]
    fn test_classify_orphan() {
        assert_eq!(classify_node(0, 0, 0, 0.0, 0.0, 5.0), Role::Orphan);
    }

    #[test]
    fn test_classify_bridge() {
        assert_eq!(classify_node(4, 2, 2, 0.5, 0.95, 5.0), Role::Bridge);
    }

    #[test]
    fn test_classify_utility() {
        // Strong sink: 5 in, 0 out → direction = -1.0
        assert_eq!(classify_node(5, 5, 0, 0.5, 0.5, 3.0), Role::Utility);
    }

    #[test]
    fn test_classify_entry_point() {
        // Strong source: 0 in, 5 out → direction = 1.0
        assert_eq!(classify_node(5, 0, 5, 0.5, 0.5, 3.0), Role::EntryPoint);
    }

    #[test]
    fn test_classify_hub() {
        // High degree, balanced, above median
        assert_eq!(classify_node(10, 5, 5, 0.95, 0.95, 3.0), Role::Hub);
    }
}
