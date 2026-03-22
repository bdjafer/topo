//! `near_disconnect` → triggers `near_disconnect`
//!
//! Removes intra-module edges to weaken a module's internal connectivity,
//! lowering its Fiedler value below the random-null p5 threshold.
//! This is the only mutation that REMOVES edges rather than adding them.

use std::collections::{HashMap, HashSet, VecDeque};

use topo_analyzer::types::{AnalysisOutput, AnalyzerInput, EdgeEntry};

use super::helpers::{clone_input, eligible_modules, flagged_nodes, remove_edges, Rng};
use super::types::{MutationResult, MutationType};

pub fn mutate(
    input: &AnalyzerInput,
    analysis: &AnalysisOutput,
    severity: u8,
    seed: u64,
) -> Option<MutationResult> {
    let mut rng = Rng::new(seed);
    let flagged = flagged_nodes(analysis, "near_disconnect");

    // Need modules with ≥ 8 members (detector requires ≥ 5, and we need
    // room to remove edges without fully disconnecting).
    let modules = eligible_modules(analysis, 8);
    if modules.is_empty() {
        return None;
    }

    // Pick a module that is NOT already flagged.
    let candidates: Vec<_> = modules
        .iter()
        .filter(|m| {
            let members: HashSet<_> = m.members.iter().cloned().collect();
            members.is_disjoint(&flagged)
        })
        .collect();

    if candidates.is_empty() {
        return None;
    }

    let candidate_ids: Vec<usize> = candidates.iter().map(|m| m.id).collect();
    let target_module = *rng.choice(&candidate_ids);
    let target_mod = analysis
        .architecture
        .modules
        .iter()
        .find(|m| m.id == target_module)?;

    let member_set: HashSet<String> = target_mod.members.iter().cloned().collect();

    // Find all intra-module edges (non-defines).
    let intra_edges: Vec<EdgeEntry> = input
        .edges
        .iter()
        .filter(|e| {
            e.kind != "defines"
                && member_set.contains(&e.source)
                && member_set.contains(&e.target)
        })
        .cloned()
        .collect();

    if intra_edges.len() < 4 {
        return None; // Too few edges to meaningfully remove.
    }

    // How many edges to remove while keeping the module connected.
    // We use a strategy: compute a spanning tree, then we know which edges
    // are tree edges (critical) and which are non-tree edges (safe to remove).
    let adj = build_undirected_adj(&intra_edges, &member_set);
    let tree_edges = spanning_tree(&adj, &member_set)?;
    let tree_set: HashSet<(String, String)> = tree_edges
        .iter()
        .flat_map(|(a, b)| vec![(a.clone(), b.clone()), (b.clone(), a.clone())])
        .collect();

    // Non-tree edges can be removed safely.
    let mut removable: Vec<EdgeEntry> = intra_edges
        .iter()
        .filter(|e| !tree_set.contains(&(e.source.clone(), e.target.clone())))
        .cloned()
        .collect();

    // How many to remove depends on severity.
    // After removing non-tree edges, also remove some tree edges (leaving bridges).
    let target_remaining_bridges = match severity {
        1 => 3, // Leave 3 bridge edges.
        2 => 2, // Leave 2 bridge edges.
        _ => 1, // Leave 1 bridge edge. Maximally fragile.
    };

    // First, remove all non-tree edges.
    let mut edges_to_remove: Vec<EdgeEntry> = Vec::new();
    edges_to_remove.extend(removable.drain(..));

    // Then remove tree edges, keeping only `target_remaining_bridges`.
    // The tree has |members| - 1 edges. Remove enough to leave target_remaining_bridges.
    let tree_edges_as_entries: Vec<EdgeEntry> = intra_edges
        .iter()
        .filter(|e| tree_set.contains(&(e.source.clone(), e.target.clone())))
        .cloned()
        .collect();

    let n_tree_to_remove = tree_edges_as_entries
        .len()
        .saturating_sub(target_remaining_bridges);
    if n_tree_to_remove > 0 {
        // Shuffle tree edges and remove from the end (leaves first would be ideal,
        // but random is good enough — the Fiedler value will drop regardless).
        let to_remove = rng.sample(&tree_edges_as_entries, n_tree_to_remove);
        edges_to_remove.extend(to_remove);
    }

    if edges_to_remove.is_empty() {
        return None;
    }

    let region: Vec<String> = member_set.into_iter().collect();

    let mut graph = clone_input(input);
    remove_edges(&mut graph, &edges_to_remove);

    Some(MutationResult {
        graph,
        mutation_type: MutationType::NearDisconnect,
        expected_diagnostic: "near_disconnect".to_string(),
        severity_level: severity,
        seed,
        added_edges: vec![],
        removed_edges: edges_to_remove,
        modified_region: region,
        description: format!(
            "Removed intra-module edges from '{}', leaving ~{} bridge edges",
            target_mod.label, target_remaining_bridges,
        ),
    })
}

/// Build undirected adjacency list from directed edges within a node set.
fn build_undirected_adj(
    edges: &[EdgeEntry],
    node_set: &HashSet<String>,
) -> HashMap<String, Vec<String>> {
    let mut adj: HashMap<String, Vec<String>> = HashMap::new();
    for nid in node_set {
        adj.entry(nid.clone()).or_default();
    }
    for e in edges {
        if node_set.contains(&e.source) && node_set.contains(&e.target) {
            adj.entry(e.source.clone())
                .or_default()
                .push(e.target.clone());
            adj.entry(e.target.clone())
                .or_default()
                .push(e.source.clone());
        }
    }
    adj
}

/// Compute a BFS spanning tree. Returns edges as (node_a, node_b) pairs.
/// Returns None if the subgraph is disconnected.
fn spanning_tree(
    adj: &HashMap<String, Vec<String>>,
    node_set: &HashSet<String>,
) -> Option<Vec<(String, String)>> {
    if node_set.is_empty() {
        return None;
    }

    let start = node_set.iter().min()?.clone(); // deterministic start
    let mut visited: HashSet<String> = HashSet::new();
    let mut tree: Vec<(String, String)> = Vec::new();
    let mut queue: VecDeque<String> = VecDeque::new();

    visited.insert(start.clone());
    queue.push_back(start);

    while let Some(node) = queue.pop_front() {
        if let Some(neighbors) = adj.get(&node) {
            for nbr in neighbors {
                if !visited.contains(nbr) {
                    visited.insert(nbr.clone());
                    tree.push((node.clone(), nbr.clone()));
                    queue.push_back(nbr.clone());
                }
            }
        }
    }

    // Check connectivity.
    if visited.len() == node_set.len() {
        Some(tree)
    } else {
        None // Subgraph is disconnected — can't build spanning tree.
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use topo_analyzer::types::NodeEntry;

    fn make_dense_module_graph() -> AnalyzerInput {
        let mut nodes = Vec::new();
        let mut edges = Vec::new();

        // 2 modules: one dense (10 nodes, many edges), one small.
        for i in 0..10 {
            nodes.push(NodeEntry {
                id: format!("dense.fn{}", i),
                kind: "function".to_string(),
                file: Some("dense/lib.rs".to_string()),
                line: Some(i as u32),
                line_end: None,
            });
        }
        // Dense internal connectivity.
        for i in 0..10 {
            for j in (i + 1)..10 {
                if (i + j) % 3 == 0 {
                    edges.push(EdgeEntry {
                        source: format!("dense.fn{}", i),
                        target: format!("dense.fn{}", j),
                        kind: "calls".to_string(),
                    });
                }
            }
        }

        // Second module.
        for i in 0..6 {
            nodes.push(NodeEntry {
                id: format!("other.fn{}", i),
                kind: "function".to_string(),
                file: Some("other/lib.rs".to_string()),
                line: Some(i as u32),
                line_end: None,
            });
        }
        for i in 0..5 {
            edges.push(EdgeEntry {
                source: format!("other.fn{}", i),
                target: format!("other.fn{}", i + 1),
                kind: "calls".to_string(),
            });
        }

        // One inter-module edge.
        edges.push(EdgeEntry {
            source: "dense.fn0".to_string(),
            target: "other.fn0".to_string(),
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
    fn near_disconnect_produces_valid_mutation() {
        let input = make_dense_module_graph();
        let analysis = topo_analyzer::analyze_full(&input);

        let result = mutate(&input, &analysis, 2, 42);
        if let Some(r) = result {
            assert_eq!(r.mutation_type, MutationType::NearDisconnect);
            assert!(!r.removed_edges.is_empty());
            assert!(!r.modified_region.is_empty());
            // Mutated graph should have fewer edges.
            assert!(r.graph.edges.len() < input.edges.len());
        }
    }
}
