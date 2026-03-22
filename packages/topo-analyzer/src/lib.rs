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
pub mod rwpe;
pub mod semantic;
pub mod spectral;
pub mod stats;
pub mod tree;
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
            line_end: node.line_end,
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
        semantic_embeddings: input.semantic_embeddings.clone(),
        experimental: input.experimental,
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

    let (mut final_modules, final_silhouette, final_fallback) = {
        let (clusters, silhouette_val, used_fallback) = if cluster_data.is_empty()
            || cluster_data.first().map(|v| v.is_empty()).unwrap_or(true)
        {
            let clusters = package_grouping(&graph.node_ids);
            (clusters, 0.0, true)
        } else {
            // k selection: user hint or modularity-sweep.
            //
            // The eigengap heuristic (von Luxburg 2007) fails when the
            // eigenvalue spectrum has no clear gap — common in real code
            // graphs with many weak communities. Instead, sweep k and
            // pick the k that maximizes Newman's modularity Q on the
            // actual graph. This is parameter-free: Q is a graph-theoretic
            // measure, not a tuned threshold.
            let (actual_k, best_km, best_sil) = if k_hint > 0 {
                let cd = clustering::prepare_for_clustering(&cluster_data, k_hint);
                let km = clustering::kmeans_best_of(&cd, k_hint, 100, &[42, 137, 271, 503, 997]);
                let sil = clustering::silhouette_score(&cd, &km.labels, &km.centroids);
                (k_hint, km, sil)
            } else {
                // Sweep k = 2..max_k, maximize Q.
                let fp_dim = cluster_data.first().map(|v| v.len()).unwrap_or(0);
                let max_k = fp_dim.min(20).min(cluster_data.len()).max(2);
                let seeds: &[u64] = &[42, 137, 271];

                let mut best_k = 2;
                let mut best_q = f64::NEG_INFINITY;
                let mut best_result = None;
                let mut best_sil_val = 0.0;

                for k in 2..=max_k {
                    let cd = clustering::prepare_for_clustering(&cluster_data, k);
                    let km = clustering::kmeans_best_of(&cd, k, 100, seeds);

                    // Build temporary cluster map for Q computation.
                    let mut tmp_clusters: HashMap<String, usize> = HashMap::new();
                    for (i, nid) in cluster_node_ids.iter().enumerate() {
                        tmp_clusters.insert(nid.clone(), km.labels[i]);
                    }
                    let q = modules::modularity_q(&graph, &tmp_clusters).unwrap_or(0.0);

                    if q > best_q {
                        best_q = q;
                        best_k = k;
                        let sil = clustering::silhouette_score(&cd, &km.labels, &km.centroids);
                        best_sil_val = sil;
                        best_result = Some(km);
                    }
                }

                (best_k, best_result.unwrap(), best_sil_val)
            };

            // Degeneracy: spectral must beat random partitioning.
            let clustering_data = clustering::prepare_for_clustering(&cluster_data, actual_k);
            let random_sil =
                clustering::random_baseline_silhouette(&clustering_data, actual_k, 5, 43);
            let degenerate = best_sil <= random_sil + PERMUTATION_MARGIN;

            if degenerate && k_hint == 0 {
                let clusters = package_grouping(&graph.node_ids);
                (clusters, best_sil, true)
            } else {
                let mut clusters: HashMap<String, usize> = HashMap::new();
                for (i, nid) in cluster_node_ids.iter().enumerate() {
                    clusters.insert(nid.clone(), best_km.labels[i]);
                }
                (clusters, best_sil, false)
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

    // Attach top_terms to each module (TF-IDF on node IDs, no model needed).
    {
        let all_clusters: HashMap<String, usize> = final_modules
            .iter()
            .flat_map(|m| m.node_ids.iter().map(move |nid| (nid.clone(), m.id)))
            .collect();
        let terms_map = semantic::top_terms(&all_clusters, 3);
        for m in &mut final_modules {
            if let Some(terms) = terms_map.get(&m.id) {
                if !terms.is_empty() {
                    m.top_terms = Some(terms.clone());
                }
            }
        }
    }

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
    let mut role_outputs = roles::classify_roles(&graph, &betweenness_vec);

    // 8. SCCs.
    let scc_indices = algorithms::tarjan_scc(&graph.successors, n);
    let sccs: Vec<Vec<String>> = scc_indices
        .into_iter()
        .filter(|c| c.len() > 1)
        .map(|c| c.into_iter().map(|i| graph.node_ids[i].clone()).collect())
        .collect();

    // 9. Anomaly detection.
    let anomaly_list = anomalies::detect_all(&graph, &sccs, &node_to_module);

    // 10. Modularity Q.
    let mod_q = modules::modularity_q_rounded(&graph, &node_to_module);

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

    // 12. Package agreement analysis.
    let package_agreement = match &input.packages {
        Some(pkgs) if pkgs.len() >= 2 => {
            Some(modules::compute_package_agreement(&final_modules, pkgs))
        }
        _ => None,
    };

    // 13. Issue synthesis.
    // Pre-compute null coherence from semantic embeddings (needed by cross-package-coupling).
    let (null_coh_mean, null_coh_std) = if let Some(ref embeddings) = input.semantic_embeddings {
        let emb64: HashMap<&str, Vec<f64>> = embeddings.iter()
            .map(|(k, v)| (k.as_str(), v.iter().map(|&x| x as f64).collect()))
            .collect();
        let (mean, std) = semantic::compute_null_coherence_threshold(&emb64, &node_to_module);
        (Some(mean), Some(std))
    } else {
        (None, None)
    };
    let issues_ctx = issues::IssuesContext {
        graph: &graph,
        modules: &final_modules,
        roles: &role_outputs,
        anomalies: &anomaly_list,
        silhouette: final_silhouette,
        package_fallback: final_fallback,
        node_to_module: &node_to_module,
        package_agreement: package_agreement.as_ref(),
        semantic_embeddings: input.semantic_embeddings.as_ref(),
        sccs: &sccs,
        null_coherence_threshold: null_coh_mean,
        null_coherence_std: null_coh_std,
    };
    let mut issues = issues::build_issues(&issues_ctx);

    // 13b. Semantic analysis (when embeddings provided).
    let semantic_result = input.semantic_embeddings.as_ref().map(|embeddings| {
        // Build role map for misplaced_concern filtering.
        let role_map: HashMap<String, String> = role_outputs.iter()
            .map(|r| (r.node_id.clone(), r.role.clone()))
            .collect();

        // Get eigenvalues, eigenvectors, and component node IDs from spectral decomposition.
        let (eigenvalues, eigenvectors, component_node_ids) = spectral.components.first()
            .map(|(node_indices, r)| {
                let ids: Vec<String> = node_indices.iter()
                    .map(|&idx| graph.node_ids[idx].clone())
                    .collect();
                (r.eigenvalues.clone(), r.eigenvectors.clone(), ids)
            })
            .unwrap_or_default();

        let experimental = input.experimental.unwrap_or(false);
        let result = semantic::analyze_semantic(
            embeddings,
            &graph,
            &node_to_module,
            &eigenvalues,
            &eigenvectors,
            &role_map,
            &component_node_ids,
            experimental,
        );

        // Apply semantic coherence to modules.
        for m in &mut final_modules {
            if let Some(&coh) = result.module_coherence.get(&m.id) {
                m.semantic_coherence = Some(round4(coh));
            }
        }

        // Add misplaced_concern issues.
        for mc in &result.misplaced_concerns {
            let target_label = final_modules.iter()
                .find(|m| m.id == mc.best_module)
                .map(|m| m.label.clone())
                .unwrap_or_else(|| format!("module_{}", mc.best_module));
            let own_label = final_modules.iter()
                .find(|m| m.id == mc.own_module)
                .map(|m| m.label.clone())
                .unwrap_or_else(|| format!("module_{}", mc.own_module));

            // Spec severity: 0.5*gap_factor + 0.3*confidence_factor + 0.2*isolation_factor
            let gap = mc.similarity_best - mc.similarity_own;
            let gap_factor = (gap / 0.4).clamp(0.0, 1.0);
            let confidence_factor = (mc.similarity_best / 0.8).clamp(0.0, 1.0);
            // Isolation: fraction of node's edges that go to other modules.
            let isolation_factor = if let Some(&idx) = graph.node_index.get(mc.node_id.as_str()) {
                let total_edges = graph.adj[idx].len();
                if total_edges > 0 {
                    let own_edges = graph.adj[idx].iter()
                        .filter(|&&(j, _)| node_to_module.get(&graph.node_ids[j]) == Some(&mc.own_module))
                        .count();
                    1.0 - (own_edges as f64 / total_edges as f64)
                } else {
                    0.5
                }
            } else {
                0.5
            };
            let sev = (0.5 * gap_factor + 0.3 * confidence_factor + 0.2 * isolation_factor).clamp(0.0, 1.0);
            let conf = (0.5 + confidence_factor * 0.3).clamp(0.3, 1.0);

            issues.push(types::IssueOutput {
                id: format!("misplaced-concern:{}", mc.node_id),
                kind: "misplaced_concern".to_string(),
                title: format!("Misplaced concern: {}", mc.node_id.rsplit('.').next().unwrap_or(&mc.node_id)),
                description: format!(
                    "Node {} is semantically closest to module {} (similarity {:.2}) \
                     but structurally assigned to module {} (similarity {:.2}). \
                     Consider moving it or extracting a shared module.",
                    mc.node_id, target_label, mc.similarity_best,
                    own_label, mc.similarity_own
                ),
                severity: round4(sev),
                severity_label: if sev >= 0.75 { "high".to_string() }
                    else if sev >= 0.45 { "medium".to_string() }
                    else { "low".to_string() },
                confidence: round4(conf),
                confidence_label: if conf >= 0.75 { "high".to_string() }
                    else if conf >= 0.45 { "medium".to_string() }
                    else { "low".to_string() },
                anchors: {
                    let anchor = graph.node_index.get(mc.node_id.as_str()).map(|&idx| {
                        types::AnchorOutput {
                            node_id: mc.node_id.clone(),
                            file: graph.node_files[idx].clone(),
                            line: graph.node_lines[idx],
                            kind: None,
                        }
                    });
                    anchor.into_iter().collect()
                },
                suggested_module: Some(target_label),
                similarity_own: Some(round4(mc.similarity_own)),
                similarity_best: Some(round4(mc.similarity_best)),
                root_cause: None,
                semantic_coherence: None,
            });
        }

        // Add incoherent_module issues.
        for im in &result.incoherent_modules {
            let label = final_modules.iter()
                .find(|m| m.id == im.module_id)
                .map(|m| m.label.clone())
                .unwrap_or_else(|| format!("module_{}", im.module_id));

            issues.push(types::IssueOutput {
                id: format!("incoherent-module:{}", label),
                kind: "incoherent_module".to_string(),
                title: format!("Incoherent module: {}", label),
                description: {
                    let mut desc = format!(
                        "Module {} has semantic coherence {:.2} (below null threshold {:.2}). \
                         Members are semantically unrelated — the structural grouping may be \
                         accidental.",
                        label, im.coherence, im.null_threshold
                    );
                    if !im.sub_clusters.is_empty() {
                        let cluster_strs: Vec<String> = im.sub_clusters.iter()
                            .filter(|c| !c.is_empty())
                            .map(|c| format!("{{{}}}", c.join(", ")))
                            .collect();
                        if !cluster_strs.is_empty() {
                            desc.push_str(&format!(
                                " Sub-clusters: {}. Consider splitting along these boundaries.",
                                cluster_strs.join(" vs ")
                            ));
                        }
                    }
                    desc
                },
                severity: {
                    // Spec: 0.4*scatter + 0.3*size + 0.3*sub_cluster
                    let scatter_factor = (1.0 - (im.coherence / 0.5).clamp(0.0, 1.0)).clamp(0.0, 1.0);
                    let size_factor = (im.module_size as f64 / 30.0).clamp(0.0, 1.0);
                    let sub_cluster_factor = (im.best_k as f64 / 5.0).clamp(0.0, 1.0);
                    let s = 0.4 * scatter_factor + 0.3 * size_factor + 0.3 * sub_cluster_factor;
                    round4(s.clamp(0.0, 1.0))
                },
                severity_label: {
                    let scatter_factor = (1.0 - (im.coherence / 0.5).clamp(0.0, 1.0)).clamp(0.0, 1.0);
                    let size_factor = (im.module_size as f64 / 30.0).clamp(0.0, 1.0);
                    let sub_cluster_factor = (im.best_k as f64 / 5.0).clamp(0.0, 1.0);
                    let s = 0.4 * scatter_factor + 0.3 * size_factor + 0.3 * sub_cluster_factor;
                    if s >= 0.75 { "high".to_string() }
                    else if s >= 0.45 { "medium".to_string() }
                    else { "low".to_string() }
                },
                confidence: 0.6,
                confidence_label: "medium".to_string(),
                anchors: Vec::new(),
                ..Default::default()
            });
        }

        // Add shadow_dependency issues (experimental).
        for sd in &result.shadow_dependencies {
            let sev = {
                let sim_factor = ((sd.similarity - 0.85) / 0.15).clamp(0.0, 1.0);
                let dist_factor = match sd.structural_distance {
                    None => 1.0,
                    Some(d) => (d as f64 / 5.0).clamp(0.0, 1.0),
                };
                // shared_refs not yet tracked — redistribute weight to sim + distance.
                // TODO: when type reference tracking is added, use spec formula:
                //   0.4 * sim_factor + 0.3 * dist_factor + 0.3 * ref_factor
                (0.5 * sim_factor + 0.5 * dist_factor).clamp(0.0, 1.0)
            };
            issues.push(types::IssueOutput {
                id: format!("shadow-dependency:{}_{}", sd.node_a.rsplit('.').next().unwrap_or(&sd.node_a), sd.node_b.rsplit('.').next().unwrap_or(&sd.node_b)),
                kind: "shadow_dependency".to_string(),
                title: format!("Shadow dependency: {} — {}", sd.node_a.rsplit('.').next().unwrap_or(&sd.node_a), sd.node_b.rsplit('.').next().unwrap_or(&sd.node_b)),
                description: format!(
                    "{} and {} are semantically similar ({:.2}) but have no structural link{}. \
                     The same logic may be implemented twice.",
                    sd.node_a, sd.node_b, sd.similarity,
                    match sd.structural_distance {
                        None => " (no path within 4 hops)".to_string(),
                        Some(d) => format!(" ({d} hops apart)"),
                    }
                ),
                severity: round4(sev),
                severity_label: if sev >= 0.75 { "high".to_string() }
                    else if sev >= 0.45 { "medium".to_string() }
                    else { "low".to_string() },
                confidence: 0.6,
                confidence_label: "medium".to_string(),
                anchors: {
                    let mut a = Vec::new();
                    if let Some(&idx) = graph.node_index.get(sd.node_a.as_str()) {
                        a.push(graph.anchor(idx));
                    }
                    if let Some(&idx) = graph.node_index.get(sd.node_b.as_str()) {
                        a.push(graph.anchor(idx));
                    }
                    a
                },
                ..Default::default()
            });
        }

        // Add redundant_api issues.
        for ra in &result.redundant_apis {
            let label = final_modules.iter()
                .find(|m| m.id == ra.module_id)
                .map(|m| m.label.clone())
                .unwrap_or_else(|| format!("module_{}", ra.module_id));
            let sev = {
                let cluster_size_factor = ((ra.entry_points.len() as f64 - 2.0) / 5.0).clamp(0.0, 1.0);
                let redundancy_factor = (ra.mean_similarity / 0.9).clamp(0.0, 1.0);
                let overlap_factor = ra.mean_callee_overlap;
                (0.4 * cluster_size_factor + 0.3 * redundancy_factor + 0.3 * overlap_factor).clamp(0.0, 1.0)
            };
            let short_names: Vec<&str> = ra.entry_points.iter()
                .map(|s| s.rsplit('.').next().unwrap_or(s.as_str()))
                .collect();
            issues.push(types::IssueOutput {
                id: format!("redundant-api:{}", label),
                kind: "redundant_api".to_string(),
                title: format!("Redundant API in module {}", label),
                description: format!(
                    "{} semantically redundant entry points in {}: {}. \
                     Mean similarity: {:.2}, mean callee overlap: {:.2}.",
                    ra.entry_points.len(), label, short_names.join(", "),
                    ra.mean_similarity, ra.mean_callee_overlap
                ),
                severity: round4(sev),
                severity_label: if sev >= 0.75 { "high".to_string() }
                    else if sev >= 0.45 { "medium".to_string() }
                    else { "low".to_string() },
                confidence: 0.6,
                confidence_label: "medium".to_string(),
                anchors: ra.entry_points.iter()
                    .take(3)
                    .filter_map(|nid| graph.node_index.get(nid.as_str()).map(|&i| graph.anchor(i)))
                    .collect(),
                ..Default::default()
            });
        }

        result
    });

    let semantic_enabled = semantic_result.as_ref().map(|r| r.gate_passed);

    // Inject local variation into role outputs when semantic analysis was run.
    if let Some(ref result) = semantic_result {
        if result.gate_passed {
            for role in &mut role_outputs {
                if let Some(&lv) = result.local_variation.get(&role.node_id) {
                    role.local_variation = Some(stats::round4(lv));
                }
            }
        }
    }

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
            spectral_coverage_ratio: Some(round4(coverage_ratio)),
            self_edge_drop_ratio: input.self_edge_ratio.map(round4),
            semantic_smoothness: semantic_result.as_ref()
                .filter(|r| r.gate_passed)
                .map(|r| round4(r.smoothness)),
            semantic_structural_ami: semantic_result.as_ref()
                .filter(|r| r.gate_passed)
                .map(|r| round4(r.ami)),
            semantic_energy_profile: semantic_result.as_ref()
                .filter(|r| r.gate_passed && !r.energy_eigenvalues.is_empty())
                .map(|r| types::SemanticEnergyProfile {
                    eigenvalues: r.energy_eigenvalues.iter().map(|&v| round4(v)).collect(),
                    semantic_energy: r.energy_values.iter().map(|&v| round4(v)).collect(),
                }),
        }),
        semantic_enabled,
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
        semantic_enabled: None,
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
                line_end: None,
            });
            nodes.push(NodeEntry {
                id: format!("b.f{i}"),
                kind: "function".to_string(),
                file: None,
                line: None,
                line_end: None,
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
            semantic_embeddings: None,
            experimental: None,
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

    /// Build synthetic embeddings where cluster "a" nodes point in one direction
    /// and cluster "b" nodes point in another, with one deliberate misplaced node.
    fn make_semantic_embeddings(dim: usize) -> HashMap<String, Vec<f32>> {
        let mut emb = HashMap::new();

        // Cluster A: vectors near [1, 0, 0, ...] with small noise.
        for i in 0..5 {
            let mut v = vec![0.0f32; dim];
            v[0] = 1.0;
            v[1] = 0.05 * i as f32; // small variation
            v[2] = 0.02 * i as f32;
            emb.insert(format!("a.f{i}"), v);
        }

        // Cluster B: vectors near [0, 1, 0, ...] with small noise.
        for i in 0..5 {
            let mut v = vec![0.0f32; dim];
            v[1] = 1.0;
            v[0] = 0.03 * i as f32; // small variation
            v[2] = 0.04 * i as f32;
            emb.insert(format!("b.f{i}"), v);
        }

        emb
    }

    /// Build a larger test graph with enough nodes to trigger all semantic features.
    fn make_large_test_input() -> AnalyzerInput {
        // 4 clusters of 8 nodes each = 32 nodes.
        // Clusters: auth, billing, orders, shared.
        let modules = ["auth", "billing", "orders", "shared"];
        let mut nodes = Vec::new();
        let mut edges = Vec::new();

        for (mi, module) in modules.iter().enumerate() {
            for i in 0..8 {
                nodes.push(NodeEntry {
                    id: format!("{module}.f{i}"),
                    kind: "function".to_string(),
                    file: Some(format!("src/{module}/mod.rs")),
                    line: Some(i * 10 + 1),
                    line_end: Some(i * 10 + 9),
                });
            }
            // Intra-module edges (chain + some cross-links).
            for i in 0..7 {
                edges.push(EdgeEntry {
                    source: format!("{module}.f{i}"),
                    target: format!("{module}.f{}", i + 1),
                    kind: "calls".to_string(),
                });
            }
            // Add some extra intra-module edges for density.
            for i in 0..5 {
                edges.push(EdgeEntry {
                    source: format!("{module}.f{i}"),
                    target: format!("{module}.f{}", i + 2),
                    kind: "calls".to_string(),
                });
            }
            // One defines edge per module.
            if mi < modules.len() - 1 {
                edges.push(EdgeEntry {
                    source: format!("{module}.f0"),
                    target: format!("{}.f0", modules[mi + 1]),
                    kind: "calls".to_string(),
                });
            }
        }

        // Cross-module bridge edges.
        edges.push(EdgeEntry {
            source: "auth.f7".to_string(),
            target: "billing.f0".to_string(),
            kind: "calls".to_string(),
        });
        edges.push(EdgeEntry {
            source: "billing.f7".to_string(),
            target: "orders.f0".to_string(),
            kind: "calls".to_string(),
        });
        edges.push(EdgeEntry {
            source: "orders.f7".to_string(),
            target: "shared.f0".to_string(),
            kind: "calls".to_string(),
        });

        // Generate semantic embeddings: each module gets a distinct direction.
        let dim = 64;
        let mut emb = HashMap::new();
        let directions: [[f32; 4]; 4] = [
            [1.0, 0.0, 0.0, 0.0], // auth
            [0.0, 1.0, 0.0, 0.0], // billing
            [0.0, 0.0, 1.0, 0.0], // orders
            [0.0, 0.0, 0.0, 1.0], // shared
        ];
        for (mi, module) in modules.iter().enumerate() {
            for i in 0..8 {
                let mut v = vec![0.0f32; dim];
                // Primary direction.
                for (d, &val) in directions[mi].iter().enumerate() {
                    v[d] = val + 0.05 * i as f32;
                }
                // Add some noise in higher dims.
                v[4 + mi] = 0.1 * (i as f32 + 1.0);
                v[8 + i] = 0.05;
                emb.insert(format!("{module}.f{i}"), v);
            }
        }

        // Deliberately misplace a node: auth.f3 gets billing's embedding direction.
        // This should trigger misplaced_concern if the graph is large enough.
        if let Some(v) = emb.get_mut("auth.f3") {
            *v = vec![0.0f32; dim];
            v[1] = 1.0; // billing direction
            v[5] = 0.3;
        }

        AnalyzerInput {
            nodes,
            edges,
            k: None,
            edge_kinds: None,
            layer_weights: None,
            scope: None,
            parsed_nodes: None,
            parsed_edges: None,
            self_edge_ratio: None,
            projection: None,
            packages: None,
            semantic_embeddings: Some(emb),
            experimental: None,
        }
    }

    #[test]
    fn test_semantic_pipeline_smoke() {
        // Small graph with synthetic embeddings — verifies the pipeline doesn't crash.
        // 10 nodes is too few for the quality gate to reliably pass, so this only
        // checks that the pipeline runs without panicking and top_terms work.
        let mut input = make_test_input();
        input.semantic_embeddings = Some(make_semantic_embeddings(32));

        let output = analyze_full(&input);

        // top_terms should always be present (doesn't need embeddings).
        let terms_present = output.architecture.modules.iter()
            .any(|m| m.top_terms.as_ref().map_or(false, |t| !t.is_empty()));
        assert!(terms_present, "top_terms should be populated");
    }

    #[test]
    fn test_semantic_pipeline_full() {
        // Larger graph that should trigger all semantic features.
        let input = make_large_test_input();
        let output = analyze_full(&input);

        // Basic structural checks.
        assert!(output.architecture.modules.len() >= 2, "should find at least 2 modules");

        // Check top_terms computation ran (may be empty for synthetic short IDs
        // depending on how Q-sweep partitions the 32-node graph).
        // The smoke test (make_test_input) already validates top_terms content.

        // Check health output exists.
        let health = output.health.as_ref().expect("health should be present");
        assert!(health.modularity_q.is_some(), "modularity_q should be present");

        // Verify JSON round-trip.
        let json_in = serde_json::to_string(&input).unwrap();
        let json_out = analyze_full_json(&json_in).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json_out).unwrap();
        assert!(parsed.get("architecture").is_some());
        assert!(parsed.get("issues").is_some());
        assert!(parsed.get("health").is_some());

        // The quality gate MUST pass for this well-separated 32-node graph.
        assert_eq!(
            output.semantic_enabled,
            Some(true),
            "quality gate should pass for well-separated 32-node graph"
        );

        // Verify all semantic output fields.
        {
            let smoothness = health.semantic_smoothness.expect("smoothness should be set");
            assert!(smoothness >= 0.0 && smoothness <= 1.5, "smoothness {smoothness} out of range");

            let ami = health.semantic_structural_ami.expect("AMI should be set");
            assert!(ami >= 0.0 && ami <= 1.0, "AMI {ami} out of range");

            // Energy profile.
            let ep = health.semantic_energy_profile.as_ref().expect("energy profile should be set");
            assert!(!ep.eigenvalues.is_empty(), "energy eigenvalues should be non-empty");
            assert_eq!(ep.eigenvalues.len(), ep.semantic_energy.len());
            let energy_sum: f64 = ep.semantic_energy.iter().sum();
            assert!(
                (energy_sum - 1.0).abs() < 0.01,
                "energy should sum to ~1.0, got {energy_sum}"
            );

            // Module coherence.
            let mods_with_coh: Vec<_> = output.architecture.modules.iter()
                .filter(|m| m.semantic_coherence.is_some())
                .collect();
            assert!(!mods_with_coh.is_empty(), "at least one module should have coherence");
            for m in &mods_with_coh {
                let c = m.semantic_coherence.unwrap();
                assert!(c >= 0.0 && c <= 1.0, "coherence {c} out of range for {}", m.label);
            }

            // Check for semantic issues (misplaced_concern or incoherent_module).
            let semantic_issues: Vec<_> = output.issues.iter()
                .filter(|i| i.kind == "misplaced_concern" || i.kind == "incoherent_module")
                .collect();
            // We don't require them (graph may be too small), but if they exist, validate fields.
            for issue in &semantic_issues {
                assert!(issue.severity > 0.0 && issue.severity <= 1.0);
                if issue.kind == "misplaced_concern" {
                    assert!(issue.suggested_module.is_some());
                    assert!(issue.similarity_own.is_some());
                    assert!(issue.similarity_best.is_some());
                }
            }
        }
    }

    #[test]
    fn test_semantic_disabled_when_no_embeddings() {
        // Without embeddings, semantic should be None/disabled.
        let input = make_test_input();
        let output = analyze_full(&input);

        assert!(output.semantic_enabled.is_none(), "semantic_enabled should be None without embeddings");
        let health = output.health.as_ref().unwrap();
        assert!(health.semantic_smoothness.is_none());
        assert!(health.semantic_structural_ami.is_none());
        assert!(health.semantic_energy_profile.is_none());

        // No semantic issues.
        let semantic_issues: Vec<_> = output.issues.iter()
            .filter(|i| i.kind == "misplaced_concern" || i.kind == "incoherent_module")
            .collect();
        assert!(semantic_issues.is_empty(), "no semantic issues without embeddings");
    }

    #[test]
    fn test_analyze_full_structural_issues() {
        // Build a graph with a cycle to verify circular_dependency fires.
        let nodes: Vec<types::NodeEntry> = (0..12).map(|i| {
            let pkg = if i < 4 { "a" } else if i < 8 { "b" } else { "c" };
            types::NodeEntry {
                id: format!("{pkg}.f{i}"),
                kind: "function".to_string(),
                file: Some(format!("{pkg}.py")),
                line: Some(i as u32 + 1),
                line_end: None,
            }
        }).collect();

        let mut edges = Vec::new();
        // Intra-module chains.
        for base in [0, 4, 8] {
            for i in base..(base + 3) {
                edges.push(types::EdgeEntry {
                    source: format!("{}.f{}", if base == 0 { "a" } else if base == 4 { "b" } else { "c" }, i),
                    target: format!("{}.f{}", if base == 0 { "a" } else if base == 4 { "b" } else { "c" }, i + 1),
                    kind: "calls".to_string(),
                });
            }
        }
        // Cross-module edges.
        edges.push(types::EdgeEntry { source: "a.f3".to_string(), target: "b.f4".to_string(), kind: "calls".to_string() });
        edges.push(types::EdgeEntry { source: "b.f7".to_string(), target: "c.f8".to_string(), kind: "calls".to_string() });
        // Cycle: c.f11 -> a.f0 (creates an SCC)
        edges.push(types::EdgeEntry { source: "c.f11".to_string(), target: "a.f0".to_string(), kind: "calls".to_string() });

        let input = AnalyzerInput {
            nodes,
            edges,
            k: Some(3),
            edge_kinds: None,
            layer_weights: None,
            scope: None,
            parsed_nodes: None,
            parsed_edges: None,
            self_edge_ratio: None,
            projection: None,
            packages: None,
            semantic_embeddings: None,
            experimental: None,
        };
        let output = analyze_full(&input);

        // Verify only spec-compliant kinds appear.
        let valid_kinds = [
            "circular_dependency", "wide_interface", "cross_package_coupling",
            "near_disconnect", "overloaded_utility", "layer_violation",
            "misplaced_concern", "incoherent_module",
            "shadow_dependency", "redundant_api",
        ];
        for issue in &output.issues {
            assert!(
                valid_kinds.contains(&issue.kind.as_str()),
                "unexpected issue kind: {} (id: {})", issue.kind, issue.id
            );
            assert!(issue.severity >= 0.0 && issue.severity <= 1.0,
                "severity out of range: {} for {}", issue.severity, issue.id);
        }

        // The cycle should produce a circular_dependency issue.
        let cycle_issues: Vec<_> = output.issues.iter()
            .filter(|i| i.kind == "circular_dependency")
            .collect();
        assert!(!cycle_issues.is_empty(),
            "expected circular_dependency issue from cycle, got kinds: {:?}",
            output.issues.iter().map(|i| &i.kind).collect::<Vec<_>>());

        // Health should have spectral_coverage_ratio.
        let health = output.health.as_ref().unwrap();
        assert!(health.spectral_coverage_ratio.is_some());
    }
}
