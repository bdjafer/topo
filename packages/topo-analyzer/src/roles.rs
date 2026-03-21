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
use crate::stats;
use crate::types::RoleOutput;

/// Percentile above which a metric is considered "high".
const PCT_THRESHOLD: f64 = 0.9;
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
    let degree_pcts = stats::percentile_ranks(&degree_f);
    let btw_pcts = stats::percentile_ranks(betweenness);

    // Data-adaptive thresholds (replaces MIN_HUB_GAP, MIN_DIRECTIONAL_DEGREE,
    // DIRECTION_THRESHOLD with distribution-derived values).
    let hub_fence = stats::tukey_upper_fence(&degree_f);

    let mean_deg = stats::mean(&degree_f);
    let std_deg = stats::std_dev(&degree_f);
    let min_directional_degree = (mean_deg - std_deg).max(2.0) as usize;

    let directions: Vec<f64> = (0..n)
        .filter(|&i| degrees[i] > 0)
        .map(|i| ((out_degrees[i] as f64 - in_degrees[i] as f64) / degrees[i] as f64).abs())
        .collect();
    let direction_fence = stats::tukey_upper_fence(&directions).clamp(0.5, 0.9);

    let mut roles = Vec::with_capacity(n);
    for i in 0..n {
        let full_degree = graph.full_in_degrees[i] + graph.full_out_degrees[i];
        let role = classify_node(
            degrees[i],
            in_degrees[i],
            out_degrees[i],
            degree_pcts[i],
            btw_pcts[i],
            betweenness[i],
            hub_fence,
            min_directional_degree,
            direction_fence,
            full_degree,
        );
        roles.push(RoleOutput {
            node_id: graph.node_ids[i].clone(),
            role: role.as_str().to_string(),
            degree: degrees[i],
            betweenness: round4(betweenness[i]),
            in_degree: in_degrees[i],
            out_degree: out_degrees[i],
            anchor: Some(graph.anchor(i)),
            local_variation: None, // Set later by semantic analysis.
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
    betweenness_raw: f64,
    hub_fence: f64,
    min_directional_degree: usize,
    direction_fence: f64,
    full_degree: usize,
) -> Role {
    // Orphan: no edges across ANY kind (including CONTAINS).
    if full_degree == 0 {
        return Role::Orphan;
    }
    // Isolated in analysis layers but connected structurally — treat as regular.
    if degree == 0 {
        return Role::Regular;
    }

    let direction = (out_degree as f64 - in_degree as f64) / degree as f64;

    // Bridge: high betweenness but not high degree.
    if betweenness_raw > 0.0
        && betweenness_pct >= PCT_THRESHOLD
        && degree_pct < PCT_THRESHOLD
    {
        return Role::Bridge;
    }

    // Directional roles: strong flow imbalance with enough edges.
    if degree >= min_directional_degree && direction <= -direction_fence {
        return Role::Utility;
    }
    if degree >= min_directional_degree && direction >= direction_fence {
        return Role::EntryPoint;
    }

    // Hub: top-percentile degree, above Tukey fence, balanced flow.
    if degree_pct >= PCT_THRESHOLD
        && degree as f64 >= hub_fence
        && direction.abs() < direction_fence
    {
        return Role::Hub;
    }

    Role::Regular
}

// Re-export for backward compatibility.
pub use crate::stats::percentile_ranks;

fn round4(v: f64) -> f64 {
    crate::stats::round4(v)
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

    // Test helper: classify_node(degree, in, out, deg_pct, btw_pct, btw_raw,
    //   hub_fence, min_dir_degree, dir_fence, full_degree)

    #[test]
    fn test_classify_orphan() {
        assert_eq!(classify_node(0, 0, 0, 0.0, 0.0, 0.0, 5.0, 3, 0.6, 0), Role::Orphan);
    }

    #[test]
    fn test_classify_structurally_connected_not_orphan() {
        assert_eq!(classify_node(0, 0, 0, 0.0, 0.0, 0.0, 5.0, 3, 0.6, 2), Role::Regular);
    }

    #[test]
    fn test_classify_bridge() {
        assert_eq!(classify_node(4, 2, 2, 0.5, 0.95, 0.05, 8.0, 3, 0.6, 4), Role::Bridge);
    }

    #[test]
    fn test_classify_bridge_rejects_zero_betweenness() {
        assert_ne!(classify_node(4, 2, 2, 0.5, 0.95, 0.0, 8.0, 3, 0.6, 4), Role::Bridge);
    }

    #[test]
    fn test_classify_utility() {
        // Strong sink: 5 in, 0 out → direction = -1.0
        assert_eq!(classify_node(5, 5, 0, 0.5, 0.5, 0.0, 8.0, 3, 0.6, 5), Role::Utility);
    }

    #[test]
    fn test_classify_entry_point() {
        // Strong source: 0 in, 5 out → direction = 1.0
        assert_eq!(classify_node(5, 0, 5, 0.5, 0.5, 0.0, 8.0, 3, 0.6, 5), Role::EntryPoint);
    }

    #[test]
    fn test_classify_hub() {
        // High degree, above fence, balanced
        assert_eq!(classify_node(10, 5, 5, 0.95, 0.95, 0.1, 8.0, 3, 0.6, 10), Role::Hub);
    }
}
