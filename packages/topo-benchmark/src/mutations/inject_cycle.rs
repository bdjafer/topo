//! `inject_cycle` → triggers `circular_dependency`
//!
//! Adds back-edges to create an SCC (strongly connected component).
//! Uses "calls" edges to avoid the all-inherits suppression rule.
//! Avoids test/spec nodes to dodge the >80% test-node suppression.

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
    let flagged = flagged_nodes(analysis, "circular_dependency");

    // Need modules with enough non-test, non-flagged members.
    let modules = eligible_modules(analysis, 3);
    if modules.is_empty() {
        return None;
    }

    let cycle_size = match severity {
        1 => 2, // 2-node cycle within one module
        2 => 3, // 3-node cycle across 2 modules
        _ => 5, // 5-node cycle across 3+ modules
    };

    let cross_module = severity >= 2;

    // Collect candidate nodes from eligible modules.
    let mut candidates: Vec<(String, usize)> = Vec::new(); // (node_id, module_id)
    for m in &modules {
        let clean = non_test_nodes(&m.members);
        for nid in clean {
            if !flagged.contains(&nid) {
                candidates.push((nid, m.id));
            }
        }
    }

    if candidates.len() < cycle_size {
        return None;
    }

    // Select nodes for the cycle.
    let selected = if cross_module {
        // Pick from different modules.
        select_cross_module(&candidates, cycle_size, &mut rng)?
    } else {
        // Pick from the same module.
        select_same_module(&candidates, cycle_size, &mut rng)?
    };

    // Create cycle edges: A→B→C→...→A
    let mut new_edges = Vec::new();
    let mut region = Vec::new();
    for i in 0..selected.len() {
        let src = &selected[i];
        let tgt = &selected[(i + 1) % selected.len()];
        new_edges.push(EdgeEntry {
            source: src.clone(),
            target: tgt.clone(),
            kind: "calls".to_string(),
        });
        region.push(src.clone());
    }

    let mut graph = clone_input(input);
    add_edges(&mut graph, &new_edges);

    Some(MutationResult {
        graph,
        mutation_type: MutationType::InjectCycle,
        expected_diagnostic: "circular_dependency".to_string(),
        severity_level: severity,
        seed,
        added_edges: new_edges,
        removed_edges: vec![],
        modified_region: region,
        description: format!(
            "Injected {}-node cycle ({}) with calls edges",
            cycle_size,
            if cross_module {
                "cross-module"
            } else {
                "intra-module"
            }
        ),
    })
}

fn select_cross_module(
    candidates: &[(String, usize)],
    k: usize,
    rng: &mut Rng,
) -> Option<Vec<String>> {
    // Group by module.
    let mut by_module: std::collections::HashMap<usize, Vec<&str>> =
        std::collections::HashMap::new();
    for (nid, mid) in candidates {
        by_module.entry(*mid).or_default().push(nid.as_str());
    }

    let module_ids: Vec<usize> = by_module.keys().copied().collect();
    if module_ids.len() < 2 {
        return None;
    }

    // Pick modules to draw from (at least 2, up to k).
    let n_modules = k.min(module_ids.len());
    let selected_modules = rng.sample(&module_ids, n_modules);

    // Round-robin pick nodes from selected modules.
    let mut result = Vec::new();
    let mut idx = 0;
    while result.len() < k {
        let mid = selected_modules[idx % selected_modules.len()];
        if let Some(nodes) = by_module.get(&mid) {
            let pick_idx = rng.next_usize(nodes.len());
            let nid = nodes[pick_idx].to_string();
            if !result.contains(&nid) {
                result.push(nid);
            }
        }
        idx += 1;
        // Safety: break if we've tried too many times.
        if idx > k * 10 {
            break;
        }
    }

    if result.len() >= k {
        result.truncate(k);
        Some(result)
    } else {
        None
    }
}

fn select_same_module(
    candidates: &[(String, usize)],
    k: usize,
    rng: &mut Rng,
) -> Option<Vec<String>> {
    // Group by module, find one with enough candidates.
    let mut by_module: std::collections::HashMap<usize, Vec<String>> =
        std::collections::HashMap::new();
    for (nid, mid) in candidates {
        by_module.entry(*mid).or_default().push(nid.clone());
    }

    let mut viable: Vec<&Vec<String>> = by_module.values().filter(|v| v.len() >= k).collect();
    if viable.is_empty() {
        return None;
    }

    // Pick a random viable module.
    let idx = rng.next_usize(viable.len());
    let nodes = viable.remove(idx);
    Some(rng.sample(nodes, k))
}

#[cfg(test)]
mod tests {
    use super::*;
    use topo_analyzer::types::AnalyzerInput;

    fn make_test_graph() -> AnalyzerInput {
        use topo_analyzer::types::NodeEntry;
        let nodes: Vec<NodeEntry> = (0..20)
            .map(|i| NodeEntry {
                id: format!("mod{}.fn{}", i / 5, i),
                kind: "function".to_string(),
                file: Some(format!("mod{}/lib.rs", i / 5)),
                line: Some(i as u32),
                line_end: None,
            })
            .collect();

        // Create a simple layered graph: mod0 → mod1 → mod2 → mod3
        let mut edges = Vec::new();
        for i in 0..16 {
            let src = format!("mod{}.fn{}", i / 5, i);
            let tgt = format!("mod{}.fn{}", (i / 5 + 1).min(3), i + 1);
            if src != tgt {
                edges.push(EdgeEntry {
                    source: src,
                    target: tgt,
                    kind: "calls".to_string(),
                });
            }
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
            semantic_embeddings: None,
            experimental: None,
        }
    }

    #[test]
    fn inject_cycle_produces_valid_mutation() {
        let input = make_test_graph();
        let analysis = topo_analyzer::analyze_full(&input);

        let result = mutate(&input, &analysis, 1, 42);
        assert!(result.is_some(), "mutation should succeed on test graph");

        let result = result.unwrap();
        assert_eq!(result.mutation_type, MutationType::InjectCycle);
        assert_eq!(result.expected_diagnostic, "circular_dependency");
        assert!(!result.added_edges.is_empty());
        assert!(!result.modified_region.is_empty());
        assert!(result.graph.edges.len() > input.edges.len());
    }

    #[test]
    fn inject_cycle_triggers_diagnostic() {
        let input = make_test_graph();
        let clean = topo_analyzer::analyze_full(&input);

        // Clean should not have circular_dependency
        let clean_has = clean.issues.iter().any(|i| i.kind == "circular_dependency");

        let result = mutate(&input, &clean, 2, 42).expect("mutation should succeed");
        let mutated = topo_analyzer::analyze_full(&result.graph);

        let mutated_has = mutated
            .issues
            .iter()
            .any(|i| i.kind == "circular_dependency");

        // If clean didn't have it, mutated should.
        if !clean_has {
            assert!(
                mutated_has,
                "inject_cycle should trigger circular_dependency. Issues: {:?}",
                mutated.issues.iter().map(|i| &i.kind).collect::<Vec<_>>()
            );
        }
    }
}
