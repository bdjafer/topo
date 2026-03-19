//! Graph construction from JSON edges → adjacency structures.

use std::collections::{HashMap, HashSet};

use crate::types::AnalyzerInput;

/// Internal graph representation: adjacency list + node index mapping.
pub struct Graph {
    /// Ordered node IDs.
    pub node_ids: Vec<String>,
    /// node_id → index.
    pub node_index: HashMap<String, usize>,
    /// Adjacency list (directed): source_index → [(target_index, weight)].
    pub adj: Vec<Vec<(usize, f64)>>,
    /// Successor list (directed, unweighted): for betweenness/SCC.
    pub successors: Vec<Vec<usize>>,
    /// Predecessor list (directed, unweighted): for betweenness.
    pub predecessors: Vec<Vec<usize>>,
    /// Number of nodes.
    pub n: usize,
}

impl Graph {
    /// Build a graph from analyzer input, filtering and weighting edges.
    pub fn from_input(input: &AnalyzerInput) -> Self {
        let node_ids: Vec<String> = input.nodes.iter().map(|n| n.id.clone()).collect();
        let node_index: HashMap<String, usize> = node_ids
            .iter()
            .enumerate()
            .map(|(i, id)| (id.clone(), i))
            .collect();
        let n = node_ids.len();

        let mut adj: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
        let mut successors: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut predecessors: Vec<Vec<usize>> = vec![Vec::new(); n];

        let allowed_kinds: Option<HashSet<&str>> = input
            .edge_kinds
            .as_ref()
            .map(|kinds| kinds.iter().map(|s| s.as_str()).collect());

        for edge in &input.edges {
            if let Some(ref allowed) = allowed_kinds {
                if !allowed.contains(edge.kind.as_str()) {
                    continue;
                }
            }
            let Some(&src) = node_index.get(&edge.source) else {
                continue;
            };
            let Some(&tgt) = node_index.get(&edge.target) else {
                continue;
            };
            if src == tgt {
                continue; // skip self-edges
            }

            let weight = input
                .layer_weights
                .as_ref()
                .and_then(|w| w.get(&edge.kind))
                .copied()
                .unwrap_or(1.0);

            adj[src].push((tgt, weight));
            successors[src].push(tgt);
            predecessors[tgt].push(src);
        }

        Graph {
            node_ids,
            node_index,
            adj,
            successors,
            predecessors,
            n,
        }
    }

    /// Build a symmetric (undirected) weighted adjacency matrix as dense f64 array.
    /// Returns row-major n×n matrix.
    pub fn symmetric_adjacency(&self) -> Vec<f64> {
        let n = self.n;
        let mut mat = vec![0.0f64; n * n];
        for (src, neighbors) in self.adj.iter().enumerate() {
            for &(tgt, w) in neighbors {
                mat[src * n + tgt] += w;
                mat[tgt * n + src] += w;
            }
        }
        mat
    }

    /// Find connected components in the undirected view of the graph.
    /// Returns a list of components, each being a list of node indices.
    pub fn connected_components(&self) -> Vec<Vec<usize>> {
        let n = self.n;
        let mut visited = vec![false; n];
        let mut components = Vec::new();

        // Build undirected adjacency for BFS.
        let mut undirected: Vec<Vec<usize>> = vec![Vec::new(); n];
        for (src, neighbors) in self.adj.iter().enumerate() {
            for &(tgt, _) in neighbors {
                undirected[src].push(tgt);
                undirected[tgt].push(src);
            }
        }

        for start in 0..n {
            if visited[start] {
                continue;
            }
            let mut component = Vec::new();
            let mut queue = std::collections::VecDeque::new();
            queue.push_back(start);
            visited[start] = true;

            while let Some(node) = queue.pop_front() {
                component.push(node);
                for &neighbor in &undirected[node] {
                    if !visited[neighbor] {
                        visited[neighbor] = true;
                        queue.push_back(neighbor);
                    }
                }
            }
            components.push(component);
        }

        // Sort by size descending.
        components.sort_by(|a, b| b.len().cmp(&a.len()));
        components
    }
}
