//! `overloaded_utility` → triggers `overloaded_utility`
//!
//! Routes calls from many modules through a single node, inflating
//! its in-degree past p85 and creating significant caller diversity.
//! Avoids nodes matching the FP suppression list (log, serialize, new, etc.).

use std::collections::HashSet;

use topo_analyzer::types::{AnalysisOutput, AnalyzerInput, EdgeEntry};

use super::helpers::{
    self, add_edges, clone_input, eligible_modules, flagged_nodes, is_suppressed_utility_name,
    non_test_nodes, Rng,
};
use super::types::{MutationResult, MutationType};

pub fn mutate(
    input: &AnalyzerInput,
    analysis: &AnalysisOutput,
    severity: u8,
    seed: u64,
) -> Option<MutationResult> {
    let mut rng = Rng::new(seed);
    let flagged = flagged_nodes(analysis, "overloaded_utility");

    let modules = eligible_modules(analysis, 3);
    if modules.len() < 3 {
        return None;
    }

    // Compute p85 threshold from in-degree distribution.
    let mut in_degrees: Vec<usize> = analysis.roles.iter().map(|r| r.in_degree).collect();
    in_degrees.sort();
    if in_degrees.is_empty() {
        return None;
    }
    let p85_idx = (in_degrees.len() as f64 * 0.85) as usize;
    let p85 = in_degrees[p85_idx.min(in_degrees.len() - 1)];

    // Pick a target node:
    // - Currently below p85 in-degree
    // - Not in the FP suppression list
    // - Not already flagged
    // - Has at least some out-degree (not a pure leaf)
    // - Not in a module named utils/helpers/common
    let node_module = helpers::node_to_module(analysis);
    // Collect candidates with their out_degree so we can prefer targets where
    // the direction significance test (in_degree > out_degree, binomial p < 0.05)
    // is easiest to satisfy. Low out_degree = fewer edges to overcome.
    let mut candidates: Vec<(String, usize)> = Vec::new(); // (node_id, out_degree)
    for role in &analysis.roles {
        if role.in_degree >= p85 {
            continue;
        }
        if flagged.contains(&role.node_id) {
            continue;
        }
        if is_suppressed_utility_name(&role.node_id) {
            continue;
        }
        if role.out_degree == 0 {
            continue;
        }
        // Skip nodes in utility-named modules (including "lib").
        if let Some(&mid) = node_module.get(&role.node_id) {
            if let Some(m) = analysis.architecture.modules.iter().find(|m| m.id == mid) {
                let label = m.label.to_lowercase();
                if label.contains("util")
                    || label.contains("helper")
                    || label.contains("common")
                    || label.contains("shared")
                    || label.contains("lib")
                {
                    continue;
                }
            }
        }
        candidates.push((role.node_id.clone(), role.out_degree));
    }

    if candidates.is_empty() {
        return None;
    }

    // Sort by out_degree ascending — targets with low out_degree are easiest
    // to push past the direction significance gate.
    candidates.sort_by_key(|(_, od)| *od);
    // Pick from the bottom quartile to keep it tractable.
    let top_n = (candidates.len() / 4).max(1);
    let pick_idx = rng.next_usize(top_n);
    let (target, target_out_degree) = candidates[pick_idx].clone();
    let target_module = node_module.get(&target).copied().unwrap_or(0);

    // Determine how many modules to route through the target.
    // We need enough incoming edges to:
    //   1. Push in_degree above p85
    //   2. Pass direction_is_significant: total_degree >= 6, in > out, binomial p < 0.05
    //   3. Exceed expected caller diversity
    // For the direction test with out_degree = d, we need in_degree >= d + 6
    // to ensure the binomial test passes reliably.
    let n_caller_modules = match severity {
        1 => (modules.len() / 2).max(3),
        2 => (modules.len() * 7 / 10).max(4),
        _ => (modules.len() * 9 / 10).max(5),
    };

    // Pick source modules (excluding the target's own module).
    let other_modules: Vec<&topo_analyzer::types::ModuleOutput> = modules
        .iter()
        .filter(|m| m.id != target_module)
        .copied()
        .collect();
    let source_modules = rng.sample(
        &other_modules.iter().map(|m| m.id).collect::<Vec<_>>(),
        n_caller_modules,
    );

    // For each source module, add calls to the target.
    // We need total added edges to exceed: p85 - current_in + target_out_degree + 6
    // to guarantee both p85 and direction significance.
    let min_total_edges = (p85 + target_out_degree + 6).max(12);
    // Use ceiling division to avoid truncation undershoot.
    let edges_per_module = match severity {
        1 => (min_total_edges.div_ceil(n_caller_modules)).max(2),
        2 => (min_total_edges.div_ceil(n_caller_modules)).max(3),
        _ => (min_total_edges.div_ceil(n_caller_modules)).max(4),
    };

    let mut new_edges = Vec::new();
    let mut region: HashSet<String> = HashSet::new();
    region.insert(target.clone());

    for mid in &source_modules {
        let m = match analysis.architecture.modules.iter().find(|m| m.id == *mid) {
            Some(m) => m,
            None => continue,
        };
        let clean_nodes = non_test_nodes(&m.members);
        if clean_nodes.is_empty() {
            continue;
        }

        let sources = rng.sample(&clean_nodes, edges_per_module);
        for src in sources {
            new_edges.push(EdgeEntry {
                source: src.clone(),
                target: target.clone(),
                kind: "calls".to_string(),
            });
            region.insert(src);
        }
    }

    if new_edges.is_empty() {
        return None;
    }

    let mut graph = clone_input(input);
    add_edges(&mut graph, &new_edges);

    Some(MutationResult {
        graph,
        mutation_type: MutationType::OverloadedUtility,
        expected_diagnostic: "overloaded_utility".to_string(),
        severity_level: severity,
        seed,
        added_edges: new_edges.clone(),
        removed_edges: vec![],
        modified_region: region.into_iter().collect(),
        description: format!(
            "Routed {} calls from {} modules to node '{}'",
            new_edges.len(),
            source_modules.len(),
            target,
        ),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use topo_analyzer::types::NodeEntry;

    fn make_test_graph() -> AnalyzerInput {
        let mut nodes = Vec::new();
        let mut edges = Vec::new();

        // 5 modules × 12 nodes = 60 nodes.
        // Dense intra-module connectivity so modules stay stable after mutation.
        // Most nodes have low in-degree (1-2), making p85 easy to exceed.
        for m in 0..5 {
            for i in 0..12 {
                nodes.push(NodeEntry {
                    id: format!("pkg{}.func{}", m, i),
                    kind: "function".to_string(),
                    file: Some(format!("pkg{}/mod.rs", m)),
                    line: Some(i as u32),
                    line_end: None,
                });
            }
            // Dense intra-module: chain + skip edges.
            for i in 0..11 {
                edges.push(EdgeEntry {
                    source: format!("pkg{}.func{}", m, i),
                    target: format!("pkg{}.func{}", m, i + 1),
                    kind: "calls".to_string(),
                });
            }
            for i in 0..10 {
                edges.push(EdgeEntry {
                    source: format!("pkg{}.func{}", m, i),
                    target: format!("pkg{}.func{}", m, i + 2),
                    kind: "calls".to_string(),
                });
            }
        }

        // Sparse one-directional inter-module edges (no return paths → no cycles).
        for m in 0..4 {
            edges.push(EdgeEntry {
                source: format!("pkg{}.func0", m),
                target: format!("pkg{}.func6", m + 1),
                kind: "calls".to_string(),
            });
        }

        AnalyzerInput {
            nodes,
            edges,
            k: Some(5),
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
    fn overloaded_utility_produces_valid_mutation() {
        let input = make_test_graph();
        let analysis = topo_analyzer::analyze_full(&input);

        let result = mutate(&input, &analysis, 2, 42);
        assert!(result.is_some(), "mutation should succeed");

        let r = result.unwrap();
        assert_eq!(r.mutation_type, MutationType::OverloadedUtility);
        assert!(!r.added_edges.is_empty());
        assert!(!r.modified_region.is_empty());
    }

    // Trigger test moved to tests/mutation_triggers.rs (uses real ripgrep graph
    // because synthetic graphs are too small for stable spectral clustering).
}
