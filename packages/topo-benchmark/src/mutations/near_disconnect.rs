//! `near_disconnect` → triggers `near_disconnect`
//!
//! Removes intra-module edges to weaken a module's internal connectivity,
//! lowering its Fiedler value below the random-null p5 threshold.
//! This is the only mutation that REMOVES edges rather than adding them.

use std::collections::HashSet;

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

    // Need modules with ≥ 6 members (detector requires ≥ 5, and we need
    // at least 1 extra node for room to remove edges).
    let modules = eligible_modules(analysis, 6);
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

    // Remove a fraction of intra-module edges to weaken internal connectivity.
    // No spanning tree needed — spectral modules often aren't internally connected,
    // so we just remove enough edges to push the Fiedler value below the null threshold.
    let remove_fraction = match severity {
        1 => 0.6, // Remove 60% of intra-module edges.
        2 => 0.75,
        _ => 0.9,
    };

    let n_to_remove = ((intra_edges.len() as f64 * remove_fraction).ceil() as usize)
        .min(intra_edges.len().saturating_sub(1)); // Keep at least 1 edge.
    let edges_to_remove = rng.sample(&intra_edges, n_to_remove);

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
            "Removed {:.0}% of intra-module edges from '{}' ({} of {} edges)",
            remove_fraction * 100.0,
            target_mod.label,
            n_to_remove,
            intra_edges.len(),
        ),
    })
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

        let result = mutate(&input, &analysis, 2, 42)
            .expect("mutation should succeed on dense module graph");
        assert_eq!(result.mutation_type, MutationType::NearDisconnect);
        assert!(!result.removed_edges.is_empty());
        assert!(!result.modified_region.is_empty());
        assert!(result.graph.edges.len() < input.edges.len());
    }

    // Trigger test moved to tests/mutation_triggers.rs (uses real ripgrep graph
    // because synthetic graphs are too small for stable spectral clustering).
}
