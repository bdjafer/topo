//! `layer_violation` → triggers `layer_violation`
//!
//! Adds "upward" edges from deeper modules to shallower modules,
//! violating the dominant directional flow. The detector requires
//! a binomial test (total ≥ 6, p < 0.05) so we add enough edges
//! to create a statistically significant minority direction.

use std::collections::{HashMap, HashSet};

use topo_analyzer::types::{AnalysisOutput, AnalyzerInput, EdgeEntry};

use super::helpers::{self, add_edges, clone_input, eligible_modules, flagged_nodes, Rng};
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

    // Find module pairs where we can add upward (reverse) edges.
    // "Upward" = from a deeper layer (higher number) to a shallower layer (lower number).
    let mut candidate_pairs: Vec<(usize, usize)> = Vec::new(); // (deep_mod, shallow_mod)
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
                    candidate_pairs.push((m_deep.id, m_shallow.id));
                }
            }
        }
    }

    if candidate_pairs.is_empty() {
        return None;
    }

    // Pick a random candidate pair.
    let (deep_id, shallow_id) = *rng.choice(&candidate_pairs);

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

    // How many upward edges to add. The detector needs total ≥ 6 and
    // the minority direction to be significant (binomial p < 0.05).
    // Adding all-reverse edges: minority = 0 forward, total = n_added.
    // Binomial(0, n, 0.5) < 0.05 for n ≥ 5. So 6 edges suffices.
    let n_edges = match severity {
        1 => 6,
        2 => 10,
        _ => 15,
    };

    // Pick source nodes from deep module, target nodes from shallow module.
    let deep_nodes = helpers::non_test_nodes(&deep_mod.members);
    let shallow_nodes = helpers::non_test_nodes(&shallow_mod.members);

    if deep_nodes.is_empty() || shallow_nodes.is_empty() {
        return None;
    }

    let mut new_edges = Vec::new();
    let mut region: HashSet<String> = HashSet::new();

    for _ in 0..n_edges {
        let src = rng.choice(&deep_nodes).clone();
        let tgt = rng.choice(&shallow_nodes).clone();
        new_edges.push(EdgeEntry {
            source: src.clone(),
            target: tgt.clone(),
            kind: "calls".to_string(),
        });
        region.insert(src);
        region.insert(tgt);
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

        // 3 modules × 6 nodes each = 18 nodes.
        // Clear layered flow: layer0 → layer1 → layer2.
        let mut nodes = Vec::new();
        let mut edges = Vec::new();

        for layer in 0..3 {
            for i in 0..6 {
                nodes.push(NodeEntry {
                    id: format!("layer{}.fn{}", layer, i),
                    kind: "function".to_string(),
                    file: Some(format!("layer{}/mod.rs", layer)),
                    line: Some(i as u32),
                    line_end: None,
                });
            }
        }

        // Forward edges: layer0 → layer1, layer1 → layer2 (8 each).
        for i in 0..4 {
            edges.push(EdgeEntry {
                source: format!("layer0.fn{}", i),
                target: format!("layer1.fn{}", i),
                kind: "calls".to_string(),
            });
            edges.push(EdgeEntry {
                source: format!("layer0.fn{}", i),
                target: format!("layer1.fn{}", i + 1),
                kind: "calls".to_string(),
            });
            edges.push(EdgeEntry {
                source: format!("layer1.fn{}", i),
                target: format!("layer2.fn{}", i),
                kind: "calls".to_string(),
            });
            edges.push(EdgeEntry {
                source: format!("layer1.fn{}", i),
                target: format!("layer2.fn{}", i + 1),
                kind: "calls".to_string(),
            });
        }

        // Intra-module edges for cohesion.
        for layer in 0..3 {
            for i in 0..5 {
                edges.push(EdgeEntry {
                    source: format!("layer{}.fn{}", layer, i),
                    target: format!("layer{}.fn{}", layer, i + 1),
                    kind: "calls".to_string(),
                });
            }
        }

        AnalyzerInput {
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
}
