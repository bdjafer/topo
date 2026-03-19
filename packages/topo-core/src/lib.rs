//! topo-core: Computational core for structural analysis.
//!
//! Provides spectral decomposition, clustering, and graph algorithms
//! as a portable library usable from Python (PyO3) and browsers (WASM).

pub mod algorithms;
pub mod clustering;
pub mod graph;
pub mod spectral;
pub mod types;

#[cfg(all(not(target_arch = "wasm32"), feature = "python"))]
pub mod python;

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub mod wasm;

use std::collections::HashMap;

use types::{AnalyzerInput, AnalyzerOutput};

/// Betweenness centrality approximation threshold (matches Python).
const BETWEENNESS_APPROX_THRESHOLD: usize = 5000;

/// Default maximum k for auto-estimation.
const DEFAULT_MAX_K: usize = 8;

/// Silhouette threshold below which clustering is degenerate.
const SILHOUETTE_DEGENERATE: f64 = 0.3;

/// Average cluster size threshold below which clustering is degenerate.
const AVG_CLUSTER_SIZE_DEGENERATE: f64 = 3.0;

/// Run the full analysis pipeline: spectral decomposition, clustering,
/// graph algorithms.
pub fn analyze(input: &AnalyzerInput) -> AnalyzerOutput {
    let graph = graph::Graph::from_input(input);
    let n = graph.n;

    if n == 0 {
        return empty_output();
    }

    // Spectral decomposition.
    let k_hint = input.k.unwrap_or(0);
    let spectral_k = if k_hint > 0 { k_hint } else { DEFAULT_MAX_K };
    let spectral = spectral::decompose(&graph, spectral_k);

    // Build fingerprint map: node_id → eigenvector coordinates.
    let mut fingerprints: HashMap<String, Vec<f64>> = HashMap::new();
    for (component_indices, result) in &spectral.components {
        for (li, &gi) in component_indices.iter().enumerate() {
            let node_id = &graph.node_ids[gi];
            fingerprints.insert(node_id.clone(), result.eigenvectors[li].clone());
        }
    }
    // Unassigned nodes get zero fingerprints.
    let fp_dim = fingerprints
        .values()
        .next()
        .map(|v| v.len())
        .unwrap_or(0);
    for component in &spectral.unassigned {
        for &gi in component {
            fingerprints.insert(graph.node_ids[gi].clone(), vec![0.0; fp_dim]);
        }
    }

    // Clustering.
    // Collect all clusterable fingerprints in order.
    let mut cluster_node_ids: Vec<String> = Vec::new();
    let mut cluster_data: Vec<Vec<f64>> = Vec::new();
    for (component_indices, result) in &spectral.components {
        for (li, &gi) in component_indices.iter().enumerate() {
            cluster_node_ids.push(graph.node_ids[gi].clone());
            cluster_data.push(result.eigenvectors[li].clone());
        }
    }

    let (clusters, silhouette, degenerate) = if cluster_data.is_empty() || cluster_data[0].is_empty() {
        // No spectral data — package grouping fallback.
        let clusters = package_grouping(&graph.node_ids);
        (clusters, 0.0, true)
    } else {
        let actual_k = if k_hint > 0 {
            k_hint
        } else {
            clustering::estimate_k(&cluster_data, DEFAULT_MAX_K, 42)
        };
        let km = clustering::kmeans(&cluster_data, actual_k, 100, 42);
        let sil = clustering::silhouette_score(&cluster_data, &km.labels, &km.centroids);

        let avg_cluster_size = cluster_data.len() as f64 / actual_k as f64;
        let degenerate = sil < SILHOUETTE_DEGENERATE || avg_cluster_size <= AVG_CLUSTER_SIZE_DEGENERATE;

        if degenerate {
            let clusters = package_grouping(&graph.node_ids);
            (clusters, sil, true)
        } else {
            let mut clusters: HashMap<String, usize> = HashMap::new();
            for (i, nid) in cluster_node_ids.iter().enumerate() {
                clusters.insert(nid.clone(), km.labels[i]);
            }
            (clusters, sil, false)
        }
    };

    // Assign unassigned nodes to a special cluster.
    let mut all_clusters = clusters;
    let max_cluster = all_clusters.values().max().copied().unwrap_or(0);
    for component in &spectral.unassigned {
        for &gi in component {
            let nid = &graph.node_ids[gi];
            if !all_clusters.contains_key(nid) {
                all_clusters.insert(nid.clone(), max_cluster + 1);
            }
        }
    }

    // Graph algorithms.
    let betweenness_vec = algorithms::brandes_betweenness(
        &graph.successors,
        n,
        BETWEENNESS_APPROX_THRESHOLD,
    );
    let mut betweenness: HashMap<String, f64> = HashMap::new();
    for (i, &b) in betweenness_vec.iter().enumerate() {
        betweenness.insert(graph.node_ids[i].clone(), b);
    }

    let scc_indices = algorithms::tarjan_scc(&graph.successors, n);
    let sccs: Vec<Vec<String>> = scc_indices
        .into_iter()
        .filter(|c| c.len() > 1)
        .map(|c| c.into_iter().map(|i| graph.node_ids[i].clone()).collect())
        .collect();

    let cc = graph.connected_components();
    let connected_components: Vec<Vec<String>> = cc
        .into_iter()
        .map(|c| c.into_iter().map(|i| graph.node_ids[i].clone()).collect())
        .collect();

    AnalyzerOutput {
        fingerprints,
        clusters: all_clusters,
        eigenvalues: spectral
            .components
            .first()
            .map(|(_, r)| r.eigenvalues.clone())
            .unwrap_or_default(),
        fiedler_value: spectral.fiedler_value,
        silhouette,
        component_sizes: spectral.component_sizes,
        betweenness,
        sccs,
        connected_components,
        degenerate,
    }
}

/// Fallback: group nodes by top-level package prefix.
fn package_grouping(node_ids: &[String]) -> HashMap<String, usize> {
    let mut package_to_cluster: HashMap<String, usize> = HashMap::new();
    let mut next_cluster = 0usize;
    let mut result: HashMap<String, usize> = HashMap::new();

    for nid in node_ids {
        let pkg = nid.split('.').next().unwrap_or(nid).to_string();
        let cluster = *package_to_cluster.entry(pkg).or_insert_with(|| {
            let c = next_cluster;
            next_cluster += 1;
            c
        });
        result.insert(nid.clone(), cluster);
    }

    result
}

fn empty_output() -> AnalyzerOutput {
    AnalyzerOutput {
        fingerprints: HashMap::new(),
        clusters: HashMap::new(),
        eigenvalues: Vec::new(),
        fiedler_value: 0.0,
        silhouette: 0.0,
        component_sizes: Vec::new(),
        betweenness: HashMap::new(),
        sccs: Vec::new(),
        connected_components: Vec::new(),
        degenerate: true,
    }
}

/// JSON entry point: takes a JSON string, returns a JSON string.
pub fn analyze_json(input_json: &str) -> Result<String, String> {
    let input: AnalyzerInput =
        serde_json::from_str(input_json).map_err(|e| format!("Invalid input: {e}"))?;
    let output = analyze(&input);
    serde_json::to_string(&output).map_err(|e| format!("Serialization error: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use types::{EdgeEntry, NodeEntry};

    fn make_test_input() -> AnalyzerInput {
        // Two clusters: a0-a4 (connected), b0-b4 (connected), one bridge edge.
        let mut nodes = Vec::new();
        let mut edges = Vec::new();

        for i in 0..5 {
            nodes.push(NodeEntry {
                id: format!("a.f{i}"),
                kind: "function".to_string(),
            });
            nodes.push(NodeEntry {
                id: format!("b.f{i}"),
                kind: "function".to_string(),
            });
        }

        // Connect within cluster a.
        for i in 0..4 {
            edges.push(EdgeEntry {
                source: format!("a.f{i}"),
                target: format!("a.f{}", i + 1),
                kind: "calls".to_string(),
            });
        }
        // Connect within cluster b.
        for i in 0..4 {
            edges.push(EdgeEntry {
                source: format!("b.f{i}"),
                target: format!("b.f{}", i + 1),
                kind: "calls".to_string(),
            });
        }
        // One bridge edge.
        edges.push(EdgeEntry {
            source: "a.f4".to_string(),
            target: "b.f0".to_string(),
            kind: "calls".to_string(),
        });

        AnalyzerInput {
            nodes,
            edges,
            k: Some(2),
            edge_kinds: None,
            layer_weights: None,
        }
    }

    #[test]
    fn test_full_pipeline() {
        let input = make_test_input();
        let output = analyze(&input);

        assert_eq!(output.fingerprints.len(), 10);
        assert_eq!(output.clusters.len(), 10);
        assert!(!output.eigenvalues.is_empty());
        assert!(output.fiedler_value >= 0.0);
        assert_eq!(output.betweenness.len(), 10);
    }

    #[test]
    fn test_json_roundtrip() {
        let input = make_test_input();
        let json_in = serde_json::to_string(&input).unwrap();
        let json_out = analyze_json(&json_in).unwrap();
        let output: AnalyzerOutput = serde_json::from_str(&json_out).unwrap();
        assert_eq!(output.fingerprints.len(), 10);
    }
}
