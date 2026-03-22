//! Structural anomaly detection.
//!
//! Detects dependency cycles via Tarjan's SCCs.

use crate::graph::Graph;
use crate::types::AnchorOutput;

/// Internal anomaly representation.
pub struct Anomaly {
    pub kind: AnomalyKind,
    pub node_ids: Vec<String>,
    pub description: String,
    pub severity: f64,
    pub confidence: f64,
    pub anchors: Vec<AnchorOutput>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnomalyKind {
    CircularDependency,
}

/// Run all anomaly detectors.
pub fn detect_all(
    graph: &Graph,
    sccs: &[Vec<String>],
    node_to_module: &std::collections::HashMap<String, usize>,
) -> Vec<Anomaly> {
    let mut anomalies = sccs_to_anomalies(sccs, graph, node_to_module);
    anomalies.sort_by(|a, b| {
        b.severity
            .partial_cmp(&a.severity)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(
                b.confidence
                    .partial_cmp(&a.confidence)
                    .unwrap_or(std::cmp::Ordering::Equal),
            )
    });
    anomalies
}

// ---------------------------------------------------------------------------
// Cycles (from pre-computed SCCs)
// ---------------------------------------------------------------------------

fn sccs_to_anomalies(
    sccs: &[Vec<String>],
    graph: &Graph,
    node_to_module: &std::collections::HashMap<String, usize>,
) -> Vec<Anomaly> {
    let non_trivial: Vec<&Vec<String>> = sccs.iter().filter(|c| c.len() > 1).collect();
    if non_trivial.is_empty() {
        return Vec::new();
    }

    // Compute p90 size for multiplier.
    let mut sizes: Vec<usize> = non_trivial.iter().map(|c| c.len()).collect();
    sizes.sort();
    let p90_idx = ((sizes.len() as f64 * 0.9) as usize).min(sizes.len().saturating_sub(1));
    let p90_size = sizes.get(p90_idx).copied().unwrap_or(usize::MAX);

    non_trivial
        .iter()
        .filter_map(|component| {
            let node_ids = {
                let mut ids = (*component).clone();
                ids.sort();
                ids
            };
            let len = node_ids.len();

            // FP suppression: test-only cycles (>80% test nodes).
            let test_count = node_ids.iter()
                .filter(|nid| {
                    let lower = nid.to_lowercase();
                    lower.contains("test") || lower.contains("spec") || lower.contains("_test")
                })
                .count();
            if test_count * 5 > len * 4 {
                // >80% test nodes → suppress.
                return None;
            }

            // FP suppression: trait/interface cycles (all edges are inherits).
            let all_inherits = {
                let scc_set: std::collections::HashSet<&str> = node_ids.iter().map(|s| s.as_str()).collect();
                let inherits_edges = graph.edges_of_kind("inherits");
                let total_scc_edges: usize = node_ids.iter()
                    .filter_map(|nid| graph.node_index.get(nid.as_str()))
                    .flat_map(|&idx| graph.adj[idx].iter().filter(|&&(tgt, _)| scc_set.contains(graph.node_ids[tgt].as_str())))
                    .count();
                let inherits_scc_edges: usize = inherits_edges.iter()
                    .filter(|&&(src, tgt)| scc_set.contains(graph.node_ids[src].as_str()) && scc_set.contains(graph.node_ids[tgt].as_str()))
                    .count();
                total_scc_edges > 0 && inherits_scc_edges == total_scc_edges
            };
            if all_inherits {
                return None;
            }

            let anchors: Vec<AnchorOutput> = node_ids
                .iter()
                .filter_map(|nid| graph.node_index.get(nid).map(|&i| graph.anchor(i)))
                .collect();

            // Spec severity: 0.4*size + 0.3*module_span + 0.3*depth
            let size_factor = (len as f64 / 20.0).clamp(0.0, 1.0);

            // Module span: distinct modules in the SCC.
            let participating_modules: std::collections::HashSet<usize> = node_ids.iter()
                .filter_map(|nid| node_to_module.get(nid).copied())
                .collect();
            let module_span_factor = (participating_modules.len() as f64 / 5.0).clamp(0.0, 1.0);

            // Depth: longest path in SCC subgraph via BFS.
            let scc_set: std::collections::HashSet<&str> = node_ids.iter().map(|s| s.as_str()).collect();
            let longest_path = {
                let mut max_depth = 0usize;
                for nid in &node_ids {
                    if let Some(&start) = graph.node_index.get(nid.as_str()) {
                        // BFS from this node within the SCC.
                        let mut visited = std::collections::HashSet::new();
                        visited.insert(start);
                        let mut queue = vec![start];
                        let mut depth = 0;
                        while !queue.is_empty() {
                            let mut next = Vec::new();
                            for &cur in &queue {
                                for &(tgt, _) in &graph.adj[cur] {
                                    if scc_set.contains(graph.node_ids[tgt].as_str())
                                        && visited.insert(tgt)
                                    {
                                        next.push(tgt);
                                    }
                                }
                            }
                            if !next.is_empty() {
                                depth += 1;
                            }
                            queue = next;
                        }
                        max_depth = max_depth.max(depth);
                    }
                }
                max_depth
            };
            let depth_factor = (longest_path as f64 / 10.0).clamp(0.0, 1.0);

            let mut sev = 0.4 * size_factor + 0.3 * module_span_factor + 0.3 * depth_factor;

            // Multipliers.
            if participating_modules.len() > 2 {
                // SCC spans >2 declared packages.
                let packages: std::collections::HashSet<&str> = node_ids.iter()
                    .map(|nid| nid.split('.').next().unwrap_or(nid.as_str()))
                    .collect();
                if packages.len() > 2 {
                    sev *= 1.4;
                }
            }
            if len > p90_size {
                sev *= 1.2;
            }

            // FP suppression: trivial 2-node same-package cycles → demote 0.5x.
            if len == 2 {
                let packages: std::collections::HashSet<&str> = node_ids.iter()
                    .map(|nid| nid.split('.').next().unwrap_or(nid.as_str()))
                    .collect();
                if packages.len() <= 1 {
                    sev *= 0.5;
                }
            }

            sev = sev.clamp(0.0, 1.0);

            Some(Anomaly {
                kind: AnomalyKind::CircularDependency,
                node_ids,
                description: format!(
                    "{} nodes form a dependency cycle spanning {} module(s). Longest path: {} hops.",
                    len, participating_modules.len(), longest_path
                ),
                severity: (sev * 100.0).round() / 100.0, // round2
                confidence: 0.9,
                anchors,
            })
        })
        .collect()
}
