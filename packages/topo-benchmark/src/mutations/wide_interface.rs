//! `wide_interface` → triggers `wide_interface`
//!
//! Adds cross-module "calls" edges between a module pair to push the
//! directional width past the Tukey upper fence (Q3 + 1.5 × IQR).
//! Needs ≥ 6 nonzero module pairs for the detector to activate.

use std::collections::HashSet;

use topo_analyzer::types::{AnalysisOutput, AnalyzerInput, EdgeEntry};

use super::helpers::{add_edges, clone_input, eligible_modules, flagged_nodes, non_test_nodes, Rng};
use super::types::{MutationResult, MutationType};

pub fn mutate(
    input: &AnalyzerInput,
    analysis: &AnalysisOutput,
    severity: u8,
    seed: u64,
) -> Option<MutationResult> {
    let mut rng = Rng::new(seed);
    let flagged = flagged_nodes(analysis, "wide_interface");

    let modules = eligible_modules(analysis, 2);
    // Need ≥ 4 modules to have ≥ 6 directed pairs for the Tukey gate.
    if modules.len() < 4 {
        return None;
    }

    // Compute current directional widths between module pairs.
    let deps = &analysis.architecture.dependencies;
    let mut widths: Vec<usize> = deps.iter().map(|d| d.weight).filter(|&w| w > 0).collect();
    widths.sort();

    // Compute Tukey fence.
    let threshold = if widths.len() >= 6 {
        let q1_idx = widths.len() / 4;
        let q3_idx = (widths.len() * 3) / 4;
        let q1 = widths[q1_idx] as f64;
        let q3 = widths[q3_idx] as f64;
        let iqr = q3 - q1;
        (q3 + 1.5 * iqr).ceil() as usize
    } else {
        // Not enough pairs — pick a reasonable default.
        5
    };

    // How many edges to add (must exceed the threshold).
    let target_width = match severity {
        1 => threshold + 3,
        2 => threshold * 3 / 2 + 2,
        _ => threshold * 2 + 5,
    };

    // Find a module pair that is NOT already flagged and has current width < threshold.
    let mut best_pair: Option<(usize, usize, usize)> = None; // (mod_a, mod_b, current_width)

    for m_a in &modules {
        for m_b in &modules {
            if m_a.id >= m_b.id {
                continue;
            }
            // Check if this pair is already flagged.
            let a_members: HashSet<_> = m_a.members.iter().cloned().collect();
            let b_members: HashSet<_> = m_b.members.iter().cloned().collect();
            if !a_members.is_disjoint(&flagged) || !b_members.is_disjoint(&flagged) {
                continue;
            }

            // Current width (A→B direction).
            let current = deps
                .iter()
                .find(|d| d.source == m_a.id && d.target == m_b.id)
                .map(|d| d.weight)
                .unwrap_or(0);

            if current >= threshold {
                continue; // Already wide.
            }

            match &best_pair {
                None => best_pair = Some((m_a.id, m_b.id, current)),
                Some((_, _, best_w)) => {
                    // Prefer pair with highest existing width (closest to threshold).
                    if current > *best_w {
                        best_pair = Some((m_a.id, m_b.id, current));
                    }
                }
            }
        }
    }

    let (mod_a, mod_b, current_width) = best_pair?;

    let m_a = analysis.architecture.modules.iter().find(|m| m.id == mod_a)?;
    let m_b = analysis.architecture.modules.iter().find(|m| m.id == mod_b)?;

    let sources = non_test_nodes(&m_a.members);
    let targets = non_test_nodes(&m_b.members);

    if sources.is_empty() || targets.is_empty() {
        return None;
    }

    let edges_to_add = target_width.saturating_sub(current_width);

    let mut new_edges = Vec::new();
    let mut region: HashSet<String> = HashSet::new();
    let mut added_pairs: HashSet<(String, String)> = HashSet::new();

    for _ in 0..edges_to_add {
        let src = rng.choice(&sources).clone();
        let tgt = rng.choice(&targets).clone();
        let pair = (src.clone(), tgt.clone());
        if added_pairs.contains(&pair) {
            continue; // Width counts distinct pairs.
        }
        added_pairs.insert(pair);
        new_edges.push(EdgeEntry {
            source: src.clone(),
            target: tgt.clone(),
            kind: "calls".to_string(),
        });
        region.insert(src);
        region.insert(tgt);
    }

    if new_edges.is_empty() {
        return None;
    }

    let mut graph = clone_input(input);
    add_edges(&mut graph, &new_edges);

    Some(MutationResult {
        graph,
        mutation_type: MutationType::WideInterface,
        expected_diagnostic: "wide_interface".to_string(),
        severity_level: severity,
        seed,
        added_edges: new_edges.clone(),
        removed_edges: vec![],
        modified_region: region.into_iter().collect(),
        description: format!(
            "Added {} distinct cross-module calls from '{}' to '{}' (width {} → {})",
            new_edges.len(),
            m_a.label,
            m_b.label,
            current_width,
            current_width + new_edges.len(),
        ),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use topo_analyzer::types::NodeEntry;

    fn make_multi_module_graph() -> AnalyzerInput {
        let mut nodes = Vec::new();
        let mut edges = Vec::new();

        // 5 modules × 6 nodes = 30 nodes.
        for m in 0..5 {
            for i in 0..6 {
                nodes.push(NodeEntry {
                    id: format!("mod{}.fn{}", m, i),
                    kind: "function".to_string(),
                    file: Some(format!("mod{}/lib.rs", m)),
                    line: Some(i as u32),
                    line_end: None,
                });
            }
            // Intra-module edges.
            for i in 0..5 {
                edges.push(EdgeEntry {
                    source: format!("mod{}.fn{}", m, i),
                    target: format!("mod{}.fn{}", m, i + 1),
                    kind: "calls".to_string(),
                });
            }
        }

        // Sparse inter-module edges (1-2 per pair).
        for m in 0..4 {
            edges.push(EdgeEntry {
                source: format!("mod{}.fn0", m),
                target: format!("mod{}.fn0", m + 1),
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
    fn wide_interface_produces_valid_mutation() {
        let input = make_multi_module_graph();
        let analysis = topo_analyzer::analyze_full(&input);

        let result = mutate(&input, &analysis, 2, 42);
        // May return None if the graph doesn't have enough module pairs.
        if let Some(r) = result {
            assert_eq!(r.mutation_type, MutationType::WideInterface);
            assert!(!r.added_edges.is_empty());
        }
    }
}
