//! topo-analyzer: Structural analysis engine.
//!
//! Provides spectral decomposition, clustering, and graph algorithms
//! as a portable library usable from Python (PyO3) and browsers (WASM).

pub mod algorithms;
pub mod anomalies;
pub mod clustering;
pub mod issues;
pub mod graph;
pub mod modules;
pub mod roles;
pub mod spectral;
pub mod stats;
pub mod types;

#[cfg(all(not(target_arch = "wasm32"), feature = "python"))]
pub mod python;

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub mod wasm;

use std::collections::{HashMap, HashSet};

use stats::round4;
use types::{AnalyzerInput, AnalyzerOutput};

/// Betweenness centrality approximation threshold (matches Python).
const BETWEENNESS_APPROX_THRESHOLD: usize = 5000;

/// Maximum eigenvectors to compute (computation budget, not cluster count).
const SPECTRAL_MAX_EIGENVECTORS: usize = 20;

/// Margin by which spectral clustering must beat random to be non-degenerate.
/// The only remaining constant — has a clear meaning: "5 percentage points
/// better than random partitioning."
const PERMUTATION_MARGIN: f64 = 0.05;

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
    let spectral_k = if k_hint > 0 { k_hint } else { SPECTRAL_MAX_EIGENVECTORS };
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
        // k from eigengap (data-adaptive) or user hint.
        let actual_k = if k_hint > 0 {
            k_hint
        } else if let Some((_, first_result)) = spectral.components.first() {
            spectral::eigengap_k(&first_result.eigenvalues)
        } else {
            2
        };

        // Ng-Jordan-Weiss: truncate to k dims + row-normalize.
        let clustering_data = clustering::prepare_for_clustering(&cluster_data, actual_k);

        let km = clustering::kmeans_best_of(&clustering_data, actual_k, 100, &[42, 137, 271, 503, 997]);
        let sil = clustering::silhouette_score(&clustering_data, &km.labels, &km.centroids);

        // Degeneracy: spectral must beat random partitioning.
        let random_sil = clustering::random_baseline_silhouette(&clustering_data, actual_k, 5, 43);
        let degenerate = sil <= random_sil + PERMUTATION_MARGIN;

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
    let unassigned_cluster_id = max_cluster + 1;
    for component in &spectral.unassigned {
        for &gi in component {
            let nid = &graph.node_ids[gi];
            if !all_clusters.contains_key(nid) {
                all_clusters.insert(nid.clone(), unassigned_cluster_id);
            }
        }
    }

    // Propagate cluster assignments to defines-only isolated nodes.
    // Nodes with no coupling edges inherit their parent's cluster
    // via the defines containment tree.
    let parent_map = graph.defines_parent_map();
    for _ in 0..20 {
        let mut changed = false;
        for gi in 0..graph.n {
            let nid = &graph.node_ids[gi];
            if let Some(&c) = all_clusters.get(nid) {
                if c != unassigned_cluster_id {
                    continue;
                }
            }
            if let Some(&parent_idx) = parent_map.get(&gi) {
                let parent_nid = &graph.node_ids[parent_idx];
                if let Some(&pc) = all_clusters.get(parent_nid) {
                    if pc != unassigned_cluster_id {
                        all_clusters.insert(nid.clone(), pc);
                        changed = true;
                    }
                }
            }
        }
        if !changed {
            break;
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

/// JSON entry point (legacy): takes a JSON string, returns a JSON string.
pub fn analyze_json(input_json: &str) -> Result<String, String> {
    let input: AnalyzerInput =
        serde_json::from_str(input_json).map_err(|e| format!("Invalid input: {e}"))?;
    let output = analyze(&input);
    serde_json::to_string(&output).map_err(|e| format!("Serialization error: {e}"))
}

/// Project a raw graph according to the projection config.
///
/// Implements the same logic as Python's `build_projection`:
/// 1. Filter nodes by kind and scope
/// 2. Lift node IDs to the target level (symbol/module/package)
/// 3. Remap edges to projected IDs, dropping self-edges
/// 4. Compute coverage and self-edge ratio
fn project_input(input: &AnalyzerInput) -> AnalyzerInput {
    use std::collections::{BTreeMap, HashSet};

    let proj = input.projection.as_ref().unwrap();
    let source_kinds: HashSet<&str> = proj.source_node_kinds.iter().map(|s| s.as_str()).collect();
    let edge_kinds: HashSet<&str> = proj.edge_kinds.iter().map(|s| s.as_str()).collect();

    // 1. Filter nodes by kind and scope.
    let selected: Vec<&types::NodeEntry> = input
        .nodes
        .iter()
        .filter(|n| {
            source_kinds.contains(n.kind.as_str())
                && is_in_scope(n.file.as_deref(), &proj.scope_roots)
        })
        .collect();

    // Build module_nodes set (nodes with kind "module").
    let module_ids: HashSet<&str> = selected
        .iter()
        .filter(|n| n.kind == "module")
        .map(|n| n.id.as_str())
        .collect();

    // 2. Build projected node ID map.
    let mut raw_to_projected: BTreeMap<&str, String> = BTreeMap::new();
    for node in &selected {
        let pid = projected_node_id(&node.id, &node.kind, &proj.level, &module_ids);
        raw_to_projected.insert(&node.id, pid);
    }

    // 3. Build projected nodes (first occurrence wins for metadata).
    let mut projected_nodes: BTreeMap<String, types::NodeEntry> = BTreeMap::new();
    for node in &selected {
        let pid = raw_to_projected[node.id.as_str()].clone();
        projected_nodes.entry(pid.clone()).or_insert_with(|| types::NodeEntry {
            id: pid,
            kind: if proj.level == "symbol" {
                node.kind.clone()
            } else {
                "module".to_string()
            },
            file: node.file.clone(),
            line: node.line,
        });
    }

    // 4. Remap edges, dropping self-edges and out-of-scope edges.
    // Duplicate projected edges are kept — Graph::from_input accumulates weights.
    let mut projected_edges: Vec<types::EdgeEntry> = Vec::new();
    let mut scoped_edge_count = 0usize;
    let mut self_edge_count = 0usize;

    for edge in &input.edges {
        if !edge_kinds.contains(edge.kind.as_str()) {
            continue;
        }
        let src_proj = raw_to_projected.get(edge.source.as_str());
        let tgt_proj = raw_to_projected.get(edge.target.as_str());
        match (src_proj, tgt_proj) {
            (Some(src), Some(tgt)) => {
                scoped_edge_count += 1;
                if src == tgt {
                    self_edge_count += 1;
                    continue;
                }
                projected_edges.push(types::EdgeEntry {
                    source: src.clone(),
                    target: tgt.clone(),
                    kind: edge.kind.clone(),
                });
            }
            _ => {
                if !proj.internal_only {
                    // Would include external edges — not implemented yet
                }
            }
        }
    }

    let self_edge_ratio = if scoped_edge_count > 0 {
        self_edge_count as f64 / scoped_edge_count as f64
    } else {
        0.0
    };

    let raw_node_count = input.nodes.len();
    let raw_edge_count = input.edges.len();

    AnalyzerInput {
        nodes: projected_nodes.into_values().collect(),
        edges: projected_edges,
        k: input.k,
        edge_kinds: Some(proj.edge_kinds.clone()),
        layer_weights: input.layer_weights.clone(),
        scope: Some(types::ScopeInput {
            level: proj.level.clone(),
            edge_kinds: proj.edge_kinds.clone(),
            internal_only: proj.internal_only,
            roots: proj.scope_roots.clone(),
        }),
        parsed_nodes: Some(raw_node_count),
        parsed_edges: Some(raw_edge_count),
        self_edge_ratio: Some(self_edge_ratio),
        projection: None, // Already projected
        packages: input.packages.clone(),
    }
}

/// Check if a file is under one of the scope roots.
fn is_in_scope(file: Option<&str>, scope_roots: &[String]) -> bool {
    if scope_roots.is_empty() {
        return true;
    }
    match file {
        Some(f) => scope_roots.iter().any(|root| f.starts_with(root)),
        None => false,
    }
}

/// Map a raw node ID to the projected ID at the given level.
fn projected_node_id(
    node_id: &str,
    node_kind: &str,
    level: &str,
    module_ids: &HashSet<&str>,
) -> String {
    if level == "symbol" {
        return node_id.to_string();
    }
    let module_id = owning_module_id(node_id, node_kind, module_ids);
    if level == "module" {
        return module_id;
    }
    // Package level: take the first dotted component.
    module_id.split('.').next().unwrap_or(&module_id).to_string()
}

/// Find the owning module for a given node.
fn owning_module_id(node_id: &str, node_kind: &str, module_ids: &HashSet<&str>) -> String {
    if node_kind == "module" {
        return node_id.to_string();
    }
    let parts: Vec<&str> = node_id.split('.').collect();
    for i in (1..parts.len()).rev() {
        let candidate: String = parts[..i].join(".");
        if module_ids.contains(candidate.as_str()) {
            return candidate;
        }
    }
    parts[0].to_string()
}

/// Run the complete analysis pipeline producing schema-compliant output.
///
/// This is the new entry point that replaces the Python analyzer entirely.
/// Output matches `schemas/analysis.schema.json`.
pub fn analyze_full(input: &AnalyzerInput) -> types::AnalysisOutput {
    // If a projection config is present, project the raw graph first.
    let projected: AnalyzerInput;
    let input = if input.projection.is_some() {
        projected = project_input(input);
        &projected
    } else {
        input
    };

    let graph = graph::Graph::from_input(input);
    let n = graph.n;

    // Scope pass-through.
    let scope = match &input.scope {
        Some(s) => types::ScopeOutput {
            level: s.level.clone(),
            edge_kinds: s.edge_kinds.clone(),
            internal_only: Some(s.internal_only),
            roots: if s.roots.is_empty() { None } else { Some(s.roots.clone()) },
        },
        None => types::ScopeOutput {
            level: "symbol".to_string(),
            edge_kinds: input
                .edge_kinds
                .clone()
                .unwrap_or_else(|| vec!["calls".to_string()]),
            internal_only: None,
            roots: None,
        },
    };

    let active_edge_kinds: Vec<String> = scope.edge_kinds.clone();

    if n == 0 {
        return empty_analysis_output(scope, input);
    }

    // 1. Spectral decomposition.
    let k_hint = input.k.unwrap_or(0);
    let spectral_k = if k_hint > 0 { k_hint } else { SPECTRAL_MAX_EIGENVECTORS };
    let spectral = spectral::decompose(&graph, spectral_k);

    // 2. Build fingerprint map.
    let mut fingerprints: HashMap<String, Vec<f64>> = HashMap::new();
    for (component_indices, result) in &spectral.components {
        for (li, &gi) in component_indices.iter().enumerate() {
            fingerprints.insert(graph.node_ids[gi].clone(), result.eigenvectors[li].clone());
        }
    }
    let fp_dim = fingerprints.values().next().map(|v| v.len()).unwrap_or(0);
    for component in &spectral.unassigned {
        for &gi in component {
            fingerprints.insert(graph.node_ids[gi].clone(), vec![0.0; fp_dim]);
        }
    }

    // 3. Module assignment.
    //
    // Always run spectral clustering regardless of package count.
    // Package boundaries are the null hypothesis to compare against,
    // not the answer. The degeneracy check (spectral must beat random)
    // is the only path to package_fallback.
    let mut cluster_node_ids: Vec<String> = Vec::new();
    let mut cluster_data: Vec<Vec<f64>> = Vec::new();
    for (component_indices, result) in &spectral.components {
        for (li, &gi) in component_indices.iter().enumerate() {
            cluster_node_ids.push(graph.node_ids[gi].clone());
            cluster_data.push(result.eigenvectors[li].clone());
        }
    }

    let (final_modules, final_silhouette, final_fallback) = {
        let (clusters, silhouette_val, used_fallback) = if cluster_data.is_empty()
            || cluster_data.first().map(|v| v.is_empty()).unwrap_or(true)
        {
            let clusters = package_grouping(&graph.node_ids);
            (clusters, 0.0, true)
        } else {
            // k from eigengap (data-adaptive) or user hint.
            let actual_k = if k_hint > 0 {
                k_hint
            } else if let Some((_, first_result)) = spectral.components.first() {
                spectral::eigengap_k(&first_result.eigenvalues)
            } else {
                2
            };

            // Ng-Jordan-Weiss: truncate to k dims + row-normalize.
            let clustering_data = clustering::prepare_for_clustering(&cluster_data, actual_k);

            let km = clustering::kmeans_best_of(&clustering_data, actual_k, 100, &[42, 137, 271, 503, 997]);
            let sil = clustering::silhouette_score(&clustering_data, &km.labels, &km.centroids);

            // Degeneracy: spectral must beat random partitioning.
            let random_sil =
                clustering::random_baseline_silhouette(&clustering_data, actual_k, 5, 43);
            let degenerate = sil <= random_sil + PERMUTATION_MARGIN;

            if degenerate && k_hint == 0 {
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
        let unassigned_cluster_id = max_cluster + 1;
        let mut has_unassigned = false;
        for component in &spectral.unassigned {
            for &gi in component {
                let nid = &graph.node_ids[gi];
                if !all_clusters.contains_key(nid) {
                    all_clusters.insert(nid.clone(), unassigned_cluster_id);
                    has_unassigned = true;
                }
            }
        }

        // Propagate cluster assignments to defines-only isolated nodes.
        // Nodes with no coupling edges inherit their parent's cluster
        // via the defines containment tree.
        let parent_map = graph.defines_parent_map();
        for _ in 0..20 {
            let mut changed = false;
            for gi in 0..graph.n {
                let nid = &graph.node_ids[gi];
                if let Some(&c) = all_clusters.get(nid) {
                    if c != unassigned_cluster_id {
                        continue;
                    }
                }
                if let Some(&parent_idx) = parent_map.get(&gi) {
                    let parent_nid = &graph.node_ids[parent_idx];
                    if let Some(&pc) = all_clusters.get(parent_nid) {
                        if pc != unassigned_cluster_id {
                            all_clusters.insert(nid.clone(), pc);
                            changed = true;
                        }
                    }
                }
            }
            if !changed {
                break;
            }
        }

        // Module enrichment.
        let silhouette_opt = if used_fallback {
            None
        } else {
            Some(silhouette_val)
        };
        let unassigned_id = if has_unassigned {
            Some(unassigned_cluster_id)
        } else {
            None
        };

        let enriched = modules::annotate_modules(
            &all_clusters,
            &fingerprints,
            silhouette_val,
            unassigned_id,
        );

        (enriched, silhouette_opt, used_fallback)
    };

    // Build node_to_module map (skipping unassigned).
    let mut node_to_module: HashMap<String, usize> = HashMap::new();
    for m in &final_modules {
        if m.unassigned {
            continue;
        }
        for nid in &m.node_ids {
            node_to_module.insert(nid.clone(), m.id);
        }
    }

    // 5. Cross-module dependencies.
    let dependencies = modules::build_module_dependencies(&graph, &node_to_module);

    // 6. Betweenness centrality.
    let betweenness_vec = algorithms::brandes_betweenness(
        &graph.successors,
        n,
        BETWEENNESS_APPROX_THRESHOLD,
    );

    // 7. Role classification.
    let role_outputs = roles::classify_roles(&graph, &betweenness_vec);

    // 8. SCCs.
    let scc_indices = algorithms::tarjan_scc(&graph.successors, n);
    let sccs: Vec<Vec<String>> = scc_indices
        .into_iter()
        .filter(|c| c.len() > 1)
        .map(|c| c.into_iter().map(|i| graph.node_ids[i].clone()).collect())
        .collect();

    // 9. Anomaly detection.
    let anomaly_list = anomalies::detect_all(
        &graph,
        &final_modules,
        &fingerprints,
        &sccs,
        &active_edge_kinds,
        final_fallback,
    );

    // 10. Modularity Q.
    let mod_q = modules::modularity_q(&graph, &node_to_module);

    // 11. Compute derived metrics for issues.
    let nodes_covered = fingerprints
        .values()
        .filter(|fp| fp.iter().any(|&v| v != 0.0))
        .count();
    let coverage_ratio = if n > 0 {
        nodes_covered as f64 / n as f64
    } else {
        0.0
    };

    let clustered_modules: Vec<&modules::EnrichedModule> =
        final_modules.iter().filter(|m| !m.unassigned).collect();
    let largest_module_size = clustered_modules
        .iter()
        .map(|m| m.node_ids.len())
        .max()
        .unwrap_or(0);
    let largest_module_ratio = if n > 0 {
        largest_module_size as f64 / n as f64
    } else {
        0.0
    };

    // 12. Package agreement analysis.
    let package_agreement = match &input.packages {
        Some(pkgs) if pkgs.len() >= 2 => {
            Some(modules::compute_package_agreement(&final_modules, pkgs))
        }
        _ => None,
    };

    // 13. Issue synthesis.
    let issues_ctx = issues::IssuesContext {
        graph: &graph,
        modules: &final_modules,
        roles: &role_outputs,
        anomalies: &anomaly_list,
        silhouette: final_silhouette,
        package_fallback: final_fallback,
        spectral_coverage_ratio: coverage_ratio,
        self_edge_ratio: input.self_edge_ratio.unwrap_or(0.0),
        level: &scope.level,
        largest_module_ratio,
        node_to_module: &node_to_module,
        package_agreement: package_agreement.as_ref(),
    };
    let issues = issues::build_issues(&issues_ctx);

    // 14. Build spectral output.
    let spectral_output = if spectral.components.is_empty() && spectral.unassigned.is_empty() {
        None
    } else {
        let largest_comp_ratio = if n > 0 {
            spectral
                .component_sizes
                .first()
                .copied()
                .unwrap_or(0) as f64
                / n as f64
        } else {
            0.0
        };
        Some(types::SpectralOutput {
            fiedler_value: spectral.fiedler_value,
            eigenvalues: spectral
                .components
                .first()
                .map(|(_, r)| r.eigenvalues.clone())
                .unwrap_or_default(),
            nodes_covered,
            coverage_ratio: round4(coverage_ratio),
            components: spectral.component_sizes.len(),
            largest_component_ratio: round4(largest_comp_ratio),
        })
    };

    // 15. Assemble output.
    types::AnalysisOutput {
        scope,
        coverage: types::CoverageOutput {
            analyzed_nodes: n,
            analyzed_edges: graph.edge_count,
            parsed_nodes: input.parsed_nodes.unwrap_or(n),
            parsed_edges: input.parsed_edges.unwrap_or(graph.edge_count),
        },
        spectral: spectral_output,
        architecture: types::ArchitectureOutput {
            modules: final_modules.iter().map(|m| m.to_output()).collect(),
            dependencies,
            silhouette: final_silhouette.map(round4),
            package_fallback: final_fallback,
            package_agreement,
        },
        roles: role_outputs,
        issues,
        health: Some(types::HealthOutput {
            modularity_q: mod_q,
        }),
    }
}

/// JSON entry point (full): takes a JSON string, returns schema-compliant JSON.
pub fn analyze_full_json(input_json: &str) -> Result<String, String> {
    let input: AnalyzerInput =
        serde_json::from_str(input_json).map_err(|e| format!("Invalid input: {e}"))?;
    let output = analyze_full(&input);
    serde_json::to_string(&output).map_err(|e| format!("Serialization error: {e}"))
}

fn empty_analysis_output(scope: types::ScopeOutput, input: &AnalyzerInput) -> types::AnalysisOutput {
    types::AnalysisOutput {
        scope,
        coverage: types::CoverageOutput {
            analyzed_nodes: 0,
            analyzed_edges: 0,
            parsed_nodes: input.parsed_nodes.unwrap_or(0),
            parsed_edges: input.parsed_edges.unwrap_or(0),
        },
        spectral: None,
        architecture: types::ArchitectureOutput {
            modules: Vec::new(),
            dependencies: Vec::new(),
            silhouette: None,
            package_fallback: true,
            package_agreement: None,
        },
        roles: Vec::new(),
        issues: Vec::new(),
        health: None,
    }
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
                file: None,
                line: None,
            });
            nodes.push(NodeEntry {
                id: format!("b.f{i}"),
                kind: "function".to_string(),
                file: None,
                line: None,
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
            scope: None,
            parsed_nodes: None,
            parsed_edges: None,
            self_edge_ratio: None,
            projection: None,
            packages: None,
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
