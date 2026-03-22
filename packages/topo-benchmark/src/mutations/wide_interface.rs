//! `wide_interface` → triggers `wide_interface`
//!
//! Adds cross-module "calls" edges between a module pair to push the
//! directional width past the Tukey upper fence (Q3 + 1.5 × IQR).
//! Needs ≥ 6 nonzero module pairs for the detector to activate.

use std::collections::{HashMap, HashSet};

use topo_analyzer::types::{AnalysisOutput, AnalyzerInput, EdgeEntry};

use super::helpers::{
    add_edges, clone_input, eligible_modules, flagged_nodes, node_to_module, non_test_nodes, Rng,
};
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

    // Compute current directional widths as DISTINCT (src, tgt) symbol pairs
    // per directed module pair — matching how the detector computes width.
    let n2m = node_to_module(analysis);
    let mut pair_symbols: HashMap<(usize, usize), HashSet<(String, String)>> = HashMap::new();
    for edge in &input.edges {
        if edge.kind == "defines" {
            continue;
        }
        let src_mod = n2m.get(&edge.source).copied();
        let tgt_mod = n2m.get(&edge.target).copied();
        if let (Some(sm), Some(tm)) = (src_mod, tgt_mod) {
            if sm != tm {
                pair_symbols
                    .entry((sm, tm))
                    .or_default()
                    .insert((edge.source.clone(), edge.target.clone()));
            }
        }
    }
    let mut widths: Vec<usize> = pair_symbols
        .values()
        .map(|s| s.len())
        .filter(|&w| w > 0)
        .collect();
    widths.sort();

    // Compute Tukey fence using the same function as the detector
    // (linear interpolation for quartiles, not nearest-rank).
    let threshold = if widths.len() >= 6 {
        let widths_f64: Vec<f64> = widths.iter().map(|&w| w as f64).collect();
        topo_analyzer::stats::tukey_upper_fence(&widths_f64).ceil() as usize
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

            // Current width (A→B direction) as distinct symbol pairs.
            let current = pair_symbols
                .get(&(m_a.id, m_b.id))
                .map(|s| s.len())
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

        // 6 modules × 10 nodes = 60 nodes.
        // Need ≥ 6 nonzero directed module pairs for the Tukey gate.
        // Dense intra-module connectivity to prevent module reassignment.
        for m in 0..6 {
            for i in 0..10 {
                nodes.push(NodeEntry {
                    id: format!("mod{}.fn{}", m, i),
                    kind: "function".to_string(),
                    file: Some(format!("mod{}/lib.rs", m)),
                    line: Some(i as u32),
                    line_end: None,
                });
            }
            // Dense intra-module edges: chain + skip edges for strong cohesion.
            for i in 0..9 {
                edges.push(EdgeEntry {
                    source: format!("mod{}.fn{}", m, i),
                    target: format!("mod{}.fn{}", m, i + 1),
                    kind: "calls".to_string(),
                });
            }
            for i in 0..8 {
                edges.push(EdgeEntry {
                    source: format!("mod{}.fn{}", m, i),
                    target: format!("mod{}.fn{}", m, i + 2),
                    kind: "calls".to_string(),
                });
            }
        }

        // Sparse inter-module edges: 2 distinct pairs per directed pair.
        // Creates 8 directed pairs with width 2 each (well below any outlier fence).
        let cross_pairs = [
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 5),
            (0, 3), (1, 4), (2, 5),
        ];
        for &(a, b) in &cross_pairs {
            for i in 0..2 {
                edges.push(EdgeEntry {
                    source: format!("mod{}.fn{}", a, i),
                    target: format!("mod{}.fn{}", b, i),
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
    fn wide_interface_produces_valid_mutation() {
        let input = make_multi_module_graph();
        let analysis = topo_analyzer::analyze_full(&input);

        let result = mutate(&input, &analysis, 2, 42)
            .expect("mutation should succeed on multi-module graph");
        assert_eq!(result.mutation_type, MutationType::WideInterface);
        assert!(!result.added_edges.is_empty());
    }

    // Trigger test moved to tests/mutation_triggers.rs (uses real ripgrep graph
    // because synthetic graphs are too small for stable spectral clustering).
}
