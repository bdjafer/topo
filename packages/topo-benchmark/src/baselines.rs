//! Baseline implementations: directory partition, Louvain, degree heuristics.

use std::collections::HashMap;

use topo_analyzer::types::AnalyzerInput;

use crate::types::{BaselineAnomaly, Partition};

/// Directory partition: group by first dotted segment of node.id.
pub fn directory_partition(input: &AnalyzerInput) -> Partition {
    let mut partition = Partition::new();
    let mut label_to_id: HashMap<String, usize> = HashMap::new();
    let mut next_id = 0;
    for node in &input.nodes {
        let pkg = node.id.split('.').next().unwrap_or(&node.id).to_string();
        let id = *label_to_id.entry(pkg).or_insert_with(|| {
            let id = next_id;
            next_id += 1;
            id
        });
        partition.insert(node.id.clone(), id);
    }
    partition
}

/// Louvain community detection (greedy modularity optimization).
///
/// Simplified single-level implementation suitable for small benchmark graphs.
pub fn louvain_partition(input: &AnalyzerInput, seed: u64) -> Partition {
    let n = input.nodes.len();
    if n == 0 {
        return Partition::new();
    }

    // Build node index.
    let id_to_idx: HashMap<&str, usize> = input
        .nodes
        .iter()
        .enumerate()
        .map(|(i, n)| (n.id.as_str(), i))
        .collect();

    // Build undirected weighted adjacency.
    let mut adj: Vec<HashMap<usize, f64>> = vec![HashMap::new(); n];
    let mut total_weight = 0.0;
    for edge in &input.edges {
        if let (Some(&si), Some(&ti)) = (id_to_idx.get(edge.source.as_str()), id_to_idx.get(edge.target.as_str())) {
            if si != ti {
                *adj[si].entry(ti).or_insert(0.0) += 1.0;
                *adj[ti].entry(si).or_insert(0.0) += 1.0;
                total_weight += 2.0;
            }
        }
    }

    if total_weight == 0.0 {
        // No edges — every node in its own community.
        return input
            .nodes
            .iter()
            .enumerate()
            .map(|(i, n)| (n.id.clone(), i))
            .collect();
    }

    let m2 = total_weight; // 2 * total edges (already doubled)

    // Degree of each node.
    let degree: Vec<f64> = (0..n)
        .map(|i| adj[i].values().sum::<f64>())
        .collect();

    // Initialize: each node in its own community.
    let mut community: Vec<usize> = (0..n).collect();

    // Sum of degrees of nodes in each community.
    let mut sigma_tot: Vec<f64> = degree.clone();

    // Deterministic node order using simple xorshift.
    let mut order: Vec<usize> = (0..n).collect();
    let mut rng = seed;
    for i in (1..n).rev() {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        let j = (rng as usize) % (i + 1);
        order.swap(i, j);
    }

    for _pass in 0..20 {
        let mut moved = false;
        for &i in &order {
            let current_comm = community[i];
            let ki = degree[i];

            // Compute weight from i to each neighboring community.
            let mut comm_weights: HashMap<usize, f64> = HashMap::new();
            for (&j, &w) in &adj[i] {
                *comm_weights.entry(community[j]).or_insert(0.0) += w;
            }

            // Weight to own community.
            let ki_in = comm_weights.get(&current_comm).copied().unwrap_or(0.0);

            // Modularity delta for removing i from current community.
            let remove_delta = -ki_in / m2 + sigma_tot[current_comm] * ki / (m2 * m2);

            let mut best_delta = 0.0;
            let mut best_comm = current_comm;

            for (&comm, &ki_comm) in &comm_weights {
                if comm == current_comm {
                    continue;
                }
                // Modularity delta for adding i to this community.
                let add_delta = ki_comm / m2 - sigma_tot[comm] * ki / (m2 * m2);
                let total_delta = remove_delta + add_delta;
                if total_delta > best_delta {
                    best_delta = total_delta;
                    best_comm = comm;
                }
            }

            if best_comm != current_comm {
                // Move node i from current_comm to best_comm.
                sigma_tot[current_comm] -= ki;
                sigma_tot[best_comm] += ki;
                community[i] = best_comm;
                moved = true;
            }
        }
        if !moved {
            break;
        }
    }

    // Renumber communities to 0..k-1.
    let mut renumber: HashMap<usize, usize> = HashMap::new();
    let mut next = 0;
    let mut partition = Partition::new();
    for (i, node) in input.nodes.iter().enumerate() {
        let c = community[i];
        let id = *renumber.entry(c).or_insert_with(|| {
            let id = next;
            next += 1;
            id
        });
        partition.insert(node.id.clone(), id);
    }
    partition
}

/// Degree heuristic anomalies: SCC detection + cross-module edge counting.
pub fn degree_heuristic_anomalies(input: &AnalyzerInput) -> Vec<BaselineAnomaly> {
    let n = input.nodes.len();
    if n == 0 {
        return vec![];
    }

    let id_to_idx: HashMap<&str, usize> = input
        .nodes
        .iter()
        .enumerate()
        .map(|(i, n)| (n.id.as_str(), i))
        .collect();

    // Build directed adjacency.
    let mut successors: Vec<Vec<usize>> = vec![vec![]; n];
    for edge in &input.edges {
        if let (Some(&si), Some(&ti)) = (id_to_idx.get(edge.source.as_str()), id_to_idx.get(edge.target.as_str())) {
            if si != ti {
                successors[si].push(ti);
            }
        }
    }

    let mut anomalies = Vec::new();

    // Tarjan's SCC.
    let sccs = tarjan_scc(&successors, n);
    for scc in &sccs {
        if scc.len() > 1 {
            let node_ids: Vec<String> = scc.iter().map(|&i| input.nodes[i].id.clone()).collect();
            let severity = (scc.len() as f64 / 10.0).min(1.0);
            anomalies.push(BaselineAnomaly {
                kind: "cycle_member".to_string(),
                node_ids,
                severity,
            });
        }
    }

    // Cross-module edge counting using directory partition.
    let dir_part = directory_partition(input);
    let mut cross_counts: HashMap<(usize, usize), usize> = HashMap::new();
    for edge in &input.edges {
        if let (Some(&src_c), Some(&tgt_c)) = (dir_part.get(&edge.source), dir_part.get(&edge.target)) {
            if src_c != tgt_c {
                *cross_counts.entry((src_c, tgt_c)).or_insert(0) += 1;
            }
        }
    }
    for ((src_c, tgt_c), count) in &cross_counts {
        if cross_counts.get(&(*tgt_c, *src_c)).is_some() {
            // Bidirectional — flag as cross-module.
            let severity = (*count as f64 / 5.0).min(1.0);
            anomalies.push(BaselineAnomaly {
                kind: "cross_module".to_string(),
                node_ids: vec![format!("cluster_{src_c}"), format!("cluster_{tgt_c}")],
                severity,
            });
        }
    }

    anomalies.sort_by(|a, b| b.severity.partial_cmp(&a.severity).unwrap_or(std::cmp::Ordering::Equal));
    anomalies
}

/// Degree heuristic role classification.
pub fn degree_heuristic_roles(input: &AnalyzerInput) -> HashMap<String, String> {
    let n = input.nodes.len();
    let id_to_idx: HashMap<&str, usize> = input
        .nodes
        .iter()
        .enumerate()
        .map(|(i, n)| (n.id.as_str(), i))
        .collect();

    let mut in_deg = vec![0usize; n];
    let mut out_deg = vec![0usize; n];
    for edge in &input.edges {
        if let (Some(&si), Some(&ti)) = (id_to_idx.get(edge.source.as_str()), id_to_idx.get(edge.target.as_str())) {
            out_deg[si] += 1;
            in_deg[ti] += 1;
        }
    }

    let degrees: Vec<usize> = (0..n).map(|i| in_deg[i] + out_deg[i]).collect();
    let mut sorted_deg = degrees.clone();
    sorted_deg.sort();
    let p90 = if sorted_deg.is_empty() {
        0
    } else {
        sorted_deg[(sorted_deg.len() as f64 * 0.9) as usize]
    };

    let mut roles = HashMap::new();
    for (i, node) in input.nodes.iter().enumerate() {
        let role = if degrees[i] == 0 {
            "orphan"
        } else if degrees[i] >= p90 && p90 > 0 {
            "hub"
        } else if out_deg[i] == 0 && in_deg[i] > 0 {
            "utility"
        } else if in_deg[i] == 0 && out_deg[i] > 0 {
            "entry_point"
        } else {
            "regular"
        };
        roles.insert(node.id.clone(), role.to_string());
    }
    roles
}

// ── Tarjan's SCC ─────────────────────────────────────────────────────────────

fn tarjan_scc(successors: &[Vec<usize>], n: usize) -> Vec<Vec<usize>> {
    let mut index_counter = 0usize;
    let mut stack = Vec::new();
    let mut on_stack = vec![false; n];
    let mut indices = vec![usize::MAX; n];
    let mut lowlinks = vec![0usize; n];
    let mut result = Vec::new();

    // Iterative Tarjan.
    for start in 0..n {
        if indices[start] != usize::MAX {
            continue;
        }

        // (node, successor_index)
        let mut call_stack: Vec<(usize, usize)> = vec![(start, 0)];
        indices[start] = index_counter;
        lowlinks[start] = index_counter;
        index_counter += 1;
        stack.push(start);
        on_stack[start] = true;

        while let Some(&mut (v, ref mut si)) = call_stack.last_mut() {
            if *si < successors[v].len() {
                let w = successors[v][*si];
                *si += 1;
                if indices[w] == usize::MAX {
                    indices[w] = index_counter;
                    lowlinks[w] = index_counter;
                    index_counter += 1;
                    stack.push(w);
                    on_stack[w] = true;
                    call_stack.push((w, 0));
                } else if on_stack[w] {
                    lowlinks[v] = lowlinks[v].min(indices[w]);
                }
            } else {
                if lowlinks[v] == indices[v] {
                    let mut scc = Vec::new();
                    while let Some(w) = stack.pop() {
                        on_stack[w] = false;
                        scc.push(w);
                        if w == v {
                            break;
                        }
                    }
                    result.push(scc);
                }
                let (v_done, _) = call_stack.pop().unwrap();
                if let Some(&(parent, _)) = call_stack.last() {
                    lowlinks[parent] = lowlinks[parent].min(lowlinks[v_done]);
                }
            }
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use topo_analyzer::types::{EdgeEntry, NodeEntry};

    fn make_input(nodes: &[&str], edges: &[(&str, &str)]) -> AnalyzerInput {
        AnalyzerInput {
            nodes: nodes
                .iter()
                .map(|id| NodeEntry {
                    id: id.to_string(),
                    kind: "module".to_string(),
                    file: None,
                    line: None,
                    line_end: None,
                })
                .collect(),
            edges: edges
                .iter()
                .map(|(s, t)| EdgeEntry {
                    source: s.to_string(),
                    target: t.to_string(),
                    kind: "calls".to_string(),
                })
                .collect(),
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
        }
    }

    #[test]
    fn test_directory_partition() {
        let input = make_input(&["api.routes", "api.views", "core.service"], &[]);
        let part = directory_partition(&input);
        assert_eq!(part.get("api.routes"), part.get("api.views"));
        assert_ne!(part.get("api.routes"), part.get("core.service"));
    }

    #[test]
    fn test_louvain_two_cliques() {
        // Two disconnected cliques should produce 2 communities.
        let input = make_input(
            &["a.1", "a.2", "a.3", "b.1", "b.2", "b.3"],
            &[
                ("a.1", "a.2"), ("a.2", "a.3"), ("a.1", "a.3"),
                ("b.1", "b.2"), ("b.2", "b.3"), ("b.1", "b.3"),
            ],
        );
        let part = louvain_partition(&input, 42);
        assert_eq!(part.get("a.1"), part.get("a.2"));
        assert_eq!(part.get("b.1"), part.get("b.2"));
        assert_ne!(part.get("a.1"), part.get("b.1"));
    }

    #[test]
    fn test_scc_detection() {
        let input = make_input(
            &["a", "b", "c"],
            &[("a", "b"), ("b", "c"), ("c", "a")],
        );
        let anomalies = degree_heuristic_anomalies(&input);
        assert!(anomalies.iter().any(|a| a.kind == "cycle_member"));
    }
}
