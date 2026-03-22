//! `layer_violation` → triggers `layer_violation`
//!
//! Adds "upward" edges from deeper modules to shallower modules,
//! violating the dominant directional flow. The detector requires
//! a binomial test (total ≥ 6, p < 0.05) so we add enough edges
//! to create a statistically significant minority direction.

use std::collections::{HashMap, HashSet};

use topo_analyzer::types::{AnalysisOutput, AnalyzerInput, EdgeEntry};

use super::helpers::{self, add_edges, clone_input, eligible_modules, flagged_nodes, node_to_module, Rng};
use super::types::{MutationResult, MutationType};

pub fn mutate(
    input: &AnalyzerInput,
    analysis: &AnalysisOutput,
    severity: u8,
    seed: u64,
) -> Option<MutationResult> {
    let mut rng = Rng::new(seed);
    let flagged = flagged_nodes(analysis, "layer_violation");

    // We need at least 2 non-unassigned modules.
    let modules = eligible_modules(analysis, 2);
    if modules.len() < 2 {
        return None;
    }

    // Reconstruct module-level DAG from dependencies.
    // The majority direction of edges defines the "forward" direction.
    // We'll compute layers via topological sort on the majority DAG.
    let layers = compute_module_layers(analysis);
    if layers.is_empty() {
        return None;
    }

    // Count existing inter-module edges per directed pair to account for
    // forward edges that would dilute the binomial test.
    let n2m = node_to_module(analysis);
    let mut edge_counts: HashMap<(usize, usize), usize> = HashMap::new();
    for edge in &input.edges {
        if edge.kind == "defines" {
            continue;
        }
        let src_mod = n2m.get(&edge.source).copied();
        let tgt_mod = n2m.get(&edge.target).copied();
        if let (Some(sm), Some(tm)) = (src_mod, tgt_mod) {
            if sm != tm {
                *edge_counts.entry((sm, tm)).or_default() += 1;
            }
        }
    }

    // Find module pairs where we can add upward (reverse) edges.
    // "Upward" = from a deeper layer (higher number) to a shallower layer (lower number).
    // Prefer pairs with few existing forward edges (shallow→deep) to avoid diluting the
    // binomial test.
    let mut candidate_pairs: Vec<(usize, usize, usize)> = Vec::new(); // (deep, shallow, existing_forward)
    for m_deep in &modules {
        for m_shallow in &modules {
            if m_deep.id == m_shallow.id {
                continue;
            }
            let l_deep = layers.get(&m_deep.id).copied().unwrap_or(0);
            let l_shallow = layers.get(&m_shallow.id).copied().unwrap_or(0);
            if l_deep > l_shallow {
                // Check neither module is already flagged.
                let deep_members: HashSet<_> = m_deep.members.iter().cloned().collect();
                let shallow_members: HashSet<_> = m_shallow.members.iter().cloned().collect();
                if deep_members.is_disjoint(&flagged) && shallow_members.is_disjoint(&flagged) {
                    // Count existing forward edges (shallow→deep, the dominant direction).
                    let existing_forward = edge_counts
                        .get(&(m_shallow.id, m_deep.id))
                        .copied()
                        .unwrap_or(0);
                    candidate_pairs.push((m_deep.id, m_shallow.id, existing_forward));
                }
            }
        }
    }

    if candidate_pairs.is_empty() {
        return None;
    }

    // Filter out pairs where a forward path exists from shallow→deep (would create SCC).
    // A simple proxy: exclude pairs where forward edges exist in BOTH directions.
    let candidate_pairs: Vec<_> = candidate_pairs
        .into_iter()
        .filter(|&(deep, shallow, _)| {
            // If there are already edges deep→shallow (our reverse direction),
            // AND shallow→deep (forward direction), we'd create/expand an SCC.
            // Only keep pairs where reverse direction has no existing edges.
            let existing_reverse = edge_counts
                .get(&(deep, shallow))
                .copied()
                .unwrap_or(0);
            existing_reverse == 0
        })
        .collect();

    if candidate_pairs.is_empty() {
        return None;
    }

    // Prefer pairs with fewest existing forward edges (easiest to trigger).
    let mut candidate_pairs = candidate_pairs;
    candidate_pairs.sort_by_key(|&(_, _, fwd)| fwd);
    let (deep_id, shallow_id, existing_forward) = candidate_pairs[0];

    let deep_mod = analysis
        .architecture
        .modules
        .iter()
        .find(|m| m.id == deep_id)?;
    let shallow_mod = analysis
        .architecture
        .modules
        .iter()
        .find(|m| m.id == shallow_id)?;

    // How many upward (reverse) edges to add. The detector needs:
    //   total = existing_forward + n_reverse >= 6
    //   binomial_cdf(existing_forward, total, 0.5) < 0.05
    //
    // Verified values for binomial_cdf(k, n, 0.5) < 0.05:
    //   k=0: n >= 5  (cdf=0.031)
    //   k=1: n >= 7  (cdf=0.016)   → need n_reverse >= 6
    //   k=2: n >= 11 (cdf=0.033)   → need n_reverse >= 9
    //   k=3: n >= 14 (cdf=0.029)   → need n_reverse >= 11
    //   k=4: n >= 16 (cdf=0.038)   → need n_reverse >= 12
    //   k=5: n >= 18 (cdf=0.048)   → need n_reverse >= 13
    //
    // Safe approximation: n_reverse >= 3 * existing_forward + 6.
    let base_edges = (3 * existing_forward + 6).max(6);
    let n_edges = match severity {
        1 => base_edges,
        2 => base_edges + 4,
        _ => base_edges + 9,
    };

    // Pick source nodes from deep module, target nodes from shallow module.
    // Use a SMALL number of distinct (src, tgt) pairs to avoid triggering
    // `wide_interface` as collateral — wide_interface counts distinct pairs,
    // while layer_violation counts total edges. We add many edges between
    // few pairs, using different edge kinds to inflate the edge count.
    let deep_nodes = helpers::non_test_nodes(&deep_mod.members);
    let shallow_nodes = helpers::non_test_nodes(&shallow_mod.members);

    if deep_nodes.is_empty() || shallow_nodes.is_empty() {
        return None;
    }

    // Pick 2-3 distinct node pairs to route all edges through.
    let n_pairs = 2.min(deep_nodes.len()).min(shallow_nodes.len());
    let selected_srcs = rng.sample(&deep_nodes, n_pairs);
    let selected_tgts = rng.sample(&shallow_nodes, n_pairs);

    let mut new_edges = Vec::new();
    let mut region: HashSet<String> = HashSet::new();
    let edge_kinds = ["calls", "imports", "calls"];

    let mut added = 0;
    while added < n_edges {
        for i in 0..n_pairs {
            if added >= n_edges {
                break;
            }
            let src = &selected_srcs[i];
            let tgt = &selected_tgts[i % selected_tgts.len()];
            new_edges.push(EdgeEntry {
                source: src.clone(),
                target: tgt.clone(),
                kind: edge_kinds[added % edge_kinds.len()].to_string(),
            });
            region.insert(src.clone());
            region.insert(tgt.clone());
            added += 1;
        }
    }

    let mut graph = clone_input(input);
    add_edges(&mut graph, &new_edges);

    Some(MutationResult {
        graph,
        mutation_type: MutationType::LayerViolation,
        expected_diagnostic: "layer_violation".to_string(),
        severity_level: severity,
        seed,
        added_edges: new_edges,
        removed_edges: vec![],
        modified_region: region.into_iter().collect(),
        description: format!(
            "Added {} upward calls edges from module {} (layer {}) to module {} (layer {})",
            n_edges,
            deep_mod.label,
            layers.get(&deep_id).unwrap_or(&0),
            shallow_mod.label,
            layers.get(&shallow_id).unwrap_or(&0),
        ),
    })
}

/// Compute module layers from inter-module dependencies using majority-direction DAG.
fn compute_module_layers(analysis: &AnalysisOutput) -> HashMap<usize, usize> {
    let deps = &analysis.architecture.dependencies;

    // Collect all module IDs.
    let mut all_ids: HashSet<usize> = HashSet::new();
    for m in &analysis.architecture.modules {
        if !m.unassigned {
            all_ids.insert(m.id);
        }
    }

    // Build majority-direction DAG.
    // For each unordered pair, the direction with more edges wins.
    let mut forward: HashMap<(usize, usize), usize> = HashMap::new();
    for dep in deps {
        *forward.entry((dep.source, dep.target)).or_default() += dep.weight;
    }

    // Build DAG edges (majority direction only).
    let mut dag_succ: HashMap<usize, Vec<usize>> = HashMap::new();
    let mut in_degree: HashMap<usize, usize> = HashMap::new();
    for &id in &all_ids {
        dag_succ.entry(id).or_default();
        in_degree.entry(id).or_default();
    }

    let mut seen_pairs: HashSet<(usize, usize)> = HashSet::new();
    for &(a, b) in forward.keys() {
        let pair = if a < b { (a, b) } else { (b, a) };
        if seen_pairs.contains(&pair) {
            continue;
        }
        seen_pairs.insert(pair);

        let ab = forward.get(&(a, b)).copied().unwrap_or(0);
        let ba = forward.get(&(b, a)).copied().unwrap_or(0);
        if ab == 0 && ba == 0 {
            continue;
        }
        let (src, tgt) = if ab >= ba { (a, b) } else { (b, a) };
        dag_succ.entry(src).or_default().push(tgt);
        *in_degree.entry(tgt).or_default() += 1;
    }

    // Longest-path BFS (Kahn's topological sort).
    let mut layers: HashMap<usize, usize> = HashMap::new();
    let mut queue: Vec<usize> = all_ids
        .iter()
        .filter(|id| *in_degree.get(id).unwrap_or(&0) == 0)
        .copied()
        .collect();
    queue.sort(); // deterministic ordering

    for &id in &queue {
        layers.insert(id, 0);
    }

    let mut i = 0;
    while i < queue.len() {
        let node = queue[i];
        i += 1;
        let node_layer = layers[&node];

        if let Some(succs) = dag_succ.get(&node) {
            for &s in succs {
                let new_layer = node_layer + 1;
                let current = layers.entry(s).or_insert(0);
                if new_layer > *current {
                    *current = new_layer;
                }
                let deg = in_degree.get_mut(&s).unwrap();
                *deg -= 1;
                if *deg == 0 {
                    queue.push(s);
                }
            }
        }
    }

    layers
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_layered_graph() -> AnalyzerInput {
        use topo_analyzer::types::NodeEntry;

        // 4 modules in a TREE (not chain) so upward edges don't create SCCs.
        // Structure:
        //   root (layer 0) → branch_a (layer 1)
        //   root (layer 0) → branch_b (layer 1)
        //   branch_a (layer 1) → leaf (layer 2)
        //
        // Adding upward edges from branch_b → root doesn't create an SCC
        // because there's no forward path root → ... → branch_b → root (only root→branch_b).
        // Wait — root→branch_b exists. branch_b→root would create SCC {root, branch_b}.
        //
        // Better: use modules where only ONE direction of edges exists.
        // branch_b → leaf has no return path. So leaf → branch_b is upward
        // but {leaf, branch_b} is not in an SCC (no edge leaf→...→branch_b→...→leaf).
        //
        // Actually with the tree: root→branch_a→leaf and root→branch_b.
        // An edge from leaf→branch_b is from layer 2 to layer 1.
        // Path check: can we go branch_b→...→leaf? Only via root→branch_a→leaf,
        // but branch_b has no edge to root. So no SCC. ✓
        //
        // Use 6 modules × 6 nodes for enough mass and enough directed pairs.
        let mut nodes = Vec::new();
        let mut edges = Vec::new();
        let module_names = ["root", "svc_a", "svc_b", "repo_a", "repo_b", "infra"];

        for name in &module_names {
            for i in 0..6 {
                nodes.push(NodeEntry {
                    id: format!("{}.fn{}", name, i),
                    kind: "function".to_string(),
                    file: Some(format!("{}/mod.rs", name)),
                    line: Some(i as u32),
                    line_end: None,
                });
            }
            // Intra-module cohesion.
            for i in 0..5 {
                edges.push(EdgeEntry {
                    source: format!("{}.fn{}", name, i),
                    target: format!("{}.fn{}", name, i + 1),
                    kind: "calls".to_string(),
                });
            }
        }

        // Tree-shaped forward edges (wider fan-out, no return paths):
        // root → svc_a (4 edges)
        // root → svc_b (4 edges)
        // svc_a → repo_a (4 edges)
        // svc_b → repo_b (4 edges)
        // repo_a → infra (4 edges)
        // repo_b → infra (4 edges)
        let forward_pairs = [
            ("root", "svc_a"),
            ("root", "svc_b"),
            ("svc_a", "repo_a"),
            ("svc_b", "repo_b"),
            ("repo_a", "infra"),
            ("repo_b", "infra"),
        ];
        for (src_mod, tgt_mod) in &forward_pairs {
            for i in 0..4 {
                edges.push(EdgeEntry {
                    source: format!("{}.fn{}", src_mod, i),
                    target: format!("{}.fn{}", tgt_mod, i),
                    kind: "calls".to_string(),
                });
            }
        }

        AnalyzerInput {
            nodes,
            edges,
            k: Some(6),
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
    fn layer_violation_produces_valid_mutation() {
        let input = make_layered_graph();
        let analysis = topo_analyzer::analyze_full(&input);

        let result = mutate(&input, &analysis, 1, 42);
        assert!(result.is_some(), "mutation should succeed on layered graph");

        let r = result.unwrap();
        assert_eq!(r.mutation_type, MutationType::LayerViolation);
        assert!(r.added_edges.len() >= 6);
        assert!(!r.modified_region.is_empty());
    }

    // Trigger test moved to tests/mutation_triggers.rs (uses real ripgrep graph
    // because synthetic graphs are too small for stable spectral clustering).
}
