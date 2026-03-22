//! Containment-tree feature extraction from `defines` edges.
//!
//! Each node gets a 4-element feature vector derived from its position in the
//! defines hierarchy: `[depth, sibling_index, subtree_size, parent_subtree_size]`.

use std::collections::{HashMap, VecDeque};

use crate::graph::Graph;

/// Extract containment-tree features from defines edges.
///
/// Returns `tree_features[node_index] = [depth, sibling_index, subtree_size, parent_subtree_size]`.
/// Nodes without defines edges get `[0, 0, 0, 0]`.
pub fn compute_tree_features(graph: &Graph) -> Vec<[usize; 4]> {
    let n = graph.n;
    if n == 0 {
        return Vec::new();
    }

    let defines_edges = graph.edges_of_kind("defines");

    // If there are no defines edges, every node is uncontained.
    if defines_edges.is_empty() {
        return vec![[0, 0, 0, 0]; n];
    }

    // Step 1: Build parent→children and child→parent maps.
    let mut children: HashMap<usize, Vec<usize>> = HashMap::new();
    let mut parent: HashMap<usize, usize> = HashMap::new();

    for &(par, child) in defines_edges {
        children.entry(par).or_default().push(child);
        parent.insert(child, par);
    }

    // Step 2: Identify roots — nodes that appear as parents but not as children.
    let mut roots: Vec<usize> = Vec::new();
    for &(par, _) in defines_edges {
        if !parent.contains_key(&par) {
            roots.push(par);
        }
    }
    roots.sort_unstable();
    roots.dedup();

    // Step 3: Sort each parent's children by node_id (alphabetical) for
    // deterministic sibling_index.
    for child_list in children.values_mut() {
        child_list.sort_by(|&a, &b| graph.node_ids[a].cmp(&graph.node_ids[b]));
    }

    // Step 4: BFS from each root to compute depth.
    let mut depth = vec![0usize; n];
    let mut involved = vec![false; n]; // tracks nodes that participate in defines

    for &root in &roots {
        let mut queue = VecDeque::new();
        queue.push_back(root);
        depth[root] = 0;
        involved[root] = true;

        while let Some(node) = queue.pop_front() {
            if let Some(kids) = children.get(&node) {
                for &kid in kids {
                    depth[kid] = depth[node] + 1;
                    involved[kid] = true;
                    queue.push_back(kid);
                }
            }
        }
    }

    // Step 5: Compute sibling_index.
    let mut sibling_index = vec![0usize; n];
    for child_list in children.values() {
        for (idx, &child) in child_list.iter().enumerate() {
            sibling_index[child] = idx;
        }
    }
    // Roots get sibling_index = 0 (already the default).

    // Step 6: Post-order traversal for subtree_size.
    // subtree_size[leaf] = 0
    // subtree_size[node] = sum over children c of (1 + subtree_size[c])
    let mut subtree_size = vec![0usize; n];
    for &root in &roots {
        compute_subtree_size(root, &children, &mut subtree_size);
    }

    // Step 7: parent_subtree_size[v] = number of children of v's parent.
    // Roots get 0.
    let mut parent_subtree_size = vec![0usize; n];
    for (&child, &par) in &parent {
        if let Some(kids) = children.get(&par) {
            parent_subtree_size[child] = kids.len();
        }
    }

    // Step 8: Assemble feature vectors.
    let mut features = vec![[0usize; 4]; n];
    for i in 0..n {
        if involved[i] {
            features[i] = [depth[i], sibling_index[i], subtree_size[i], parent_subtree_size[i]];
        }
        // else: [0, 0, 0, 0] — the default
    }

    features
}

/// Recursive post-order computation of subtree sizes.
fn compute_subtree_size(
    node: usize,
    children: &HashMap<usize, Vec<usize>>,
    subtree_size: &mut [usize],
) {
    if let Some(kids) = children.get(&node) {
        let mut total = 0usize;
        for &kid in kids {
            compute_subtree_size(kid, children, subtree_size);
            total += 1 + subtree_size[kid];
        }
        subtree_size[node] = total;
    }
    // else: leaf, subtree_size stays 0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{AnalyzerInput, EdgeEntry, NodeEntry};

    fn make_test_graph(n: usize, node_ids: Vec<&str>, edges: Vec<(usize, usize, &str)>) -> Graph {
        let nodes: Vec<NodeEntry> = if node_ids.is_empty() {
            (0..n)
                .map(|i| NodeEntry {
                    id: format!("n{i}"),
                    kind: "function".to_string(),
                    file: None,
                    line: None,
                    line_end: None,
                })
                .collect()
        } else {
            node_ids
                .iter()
                .map(|id| NodeEntry {
                    id: id.to_string(),
                    kind: "function".to_string(),
                    file: None,
                    line: None,
                    line_end: None,
                })
                .collect()
        };

        let node_id_list: Vec<String> = nodes.iter().map(|n| n.id.clone()).collect();

        let edge_entries: Vec<EdgeEntry> = edges
            .iter()
            .map(|(s, t, k)| EdgeEntry {
                source: node_id_list[*s].clone(),
                target: node_id_list[*t].clone(),
                kind: k.to_string(),
            })
            .collect();

        let input = AnalyzerInput {
            nodes,
            edges: edge_entries,
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
        };
        Graph::from_input(&input)
    }

    #[test]
    fn test_three_level_tree() {
        // package → module → {fn_a, fn_b, fn_c}
        // Node IDs chosen so alphabetical sort is testable.
        let graph = make_test_graph(
            5,
            vec!["pkg", "pkg.mod", "pkg.mod.fn_a", "pkg.mod.fn_b", "pkg.mod.fn_c"],
            vec![
                (0, 1, "defines"), // pkg defines pkg.mod
                (1, 2, "defines"), // pkg.mod defines fn_a
                (1, 3, "defines"), // pkg.mod defines fn_b
                (1, 4, "defines"), // pkg.mod defines fn_c
            ],
        );

        let features = compute_tree_features(&graph);

        // pkg: depth=0, sibling_index=0, subtree_size=4, parent_subtree_size=0
        assert_eq!(features[0], [0, 0, 4, 0], "pkg features");
        // pkg.mod: depth=1, sibling_index=0, subtree_size=3, parent_subtree_size=1
        assert_eq!(features[1], [1, 0, 3, 1], "pkg.mod features");
        // pkg.mod.fn_a: depth=2, sibling_index=0, subtree_size=0, parent_subtree_size=3
        assert_eq!(features[2], [2, 0, 0, 3], "pkg.mod.fn_a features");
        // pkg.mod.fn_b: depth=2, sibling_index=1, subtree_size=0, parent_subtree_size=3
        assert_eq!(features[3], [2, 1, 0, 3], "pkg.mod.fn_b features");
        // pkg.mod.fn_c: depth=2, sibling_index=2, subtree_size=0, parent_subtree_size=3
        assert_eq!(features[4], [2, 2, 0, 3], "pkg.mod.fn_c features");
    }

    #[test]
    fn test_forest() {
        // Two independent packages, each with one child.
        let graph = make_test_graph(
            4,
            vec!["alpha", "alpha.child", "beta", "beta.child"],
            vec![
                (0, 1, "defines"), // alpha defines alpha.child
                (2, 3, "defines"), // beta defines beta.child
            ],
        );

        let features = compute_tree_features(&graph);

        // alpha: root → depth=0, subtree_size=1, parent_subtree_size=0
        assert_eq!(features[0], [0, 0, 1, 0], "alpha features");
        // alpha.child: depth=1, sibling_index=0, subtree_size=0, parent_subtree_size=1
        assert_eq!(features[1], [1, 0, 0, 1], "alpha.child features");
        // beta: root → depth=0, subtree_size=1, parent_subtree_size=0
        assert_eq!(features[2], [0, 0, 1, 0], "beta features");
        // beta.child: depth=1, sibling_index=0, subtree_size=0, parent_subtree_size=1
        assert_eq!(features[3], [1, 0, 0, 1], "beta.child features");
    }

    #[test]
    fn test_orphan_nodes() {
        // 4 nodes, only 2 are connected by defines. The other 2 are orphans.
        let graph = make_test_graph(
            4,
            vec!["mod", "mod.fn", "orphan_a", "orphan_b"],
            vec![
                (0, 1, "defines"), // mod defines mod.fn
            ],
        );

        let features = compute_tree_features(&graph);

        // mod: root
        assert_eq!(features[0], [0, 0, 1, 0], "mod features");
        // mod.fn: child
        assert_eq!(features[1], [1, 0, 0, 1], "mod.fn features");
        // orphan_a: no defines involvement → [0,0,0,0]
        assert_eq!(features[2], [0, 0, 0, 0], "orphan_a features");
        // orphan_b: no defines involvement → [0,0,0,0]
        assert_eq!(features[3], [0, 0, 0, 0], "orphan_b features");
    }

    #[test]
    fn test_single_node() {
        let graph = make_test_graph(1, vec!["lonely"], vec![]);

        let features = compute_tree_features(&graph);

        assert_eq!(features.len(), 1);
        assert_eq!(features[0], [0, 0, 0, 0], "single node features");
    }

    #[test]
    fn test_empty_graph() {
        let graph = make_test_graph(0, vec![], vec![]);

        let features = compute_tree_features(&graph);

        assert!(features.is_empty());
    }
}
