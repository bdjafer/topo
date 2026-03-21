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

    // --- Enriched metadata for full analysis ---

    /// Edges grouped by kind: kind_str → [(source_index, target_index)].
    pub typed_edges: HashMap<String, Vec<(usize, usize)>>,
    /// Node kinds (from input), indexed by node index.
    pub node_kinds: Vec<String>,
    /// Node file paths (from input), indexed by node index.
    pub node_files: Vec<Option<String>>,
    /// Node line numbers (from input), indexed by node index.
    pub node_lines: Vec<Option<u32>>,
    /// Total edge count (after filtering).
    pub edge_count: usize,
    /// In-degree across ALL edge kinds (including filtered-out ones like CONTAINS).
    /// Used for orphan detection — a node with CONTAINS edges isn't orphaned.
    pub full_in_degrees: Vec<usize>,
    /// Out-degree across ALL edge kinds.
    pub full_out_degrees: Vec<usize>,
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

        let node_kinds: Vec<String> = input.nodes.iter().map(|n| n.kind.clone()).collect();
        let node_files: Vec<Option<String>> = input.nodes.iter().map(|n| n.file.clone()).collect();
        let node_lines: Vec<Option<u32>> = input.nodes.iter().map(|n| n.line).collect();

        let mut adj: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
        let mut successors: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut predecessors: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut typed_edges: HashMap<String, Vec<(usize, usize)>> = HashMap::new();
        let mut edge_count = 0usize;

        // Full-degree arrays: count ALL edge kinds (before filtering) for orphan detection.
        let mut full_in_degrees = vec![0usize; n];
        let mut full_out_degrees = vec![0usize; n];
        for edge in &input.edges {
            let Some(&src) = node_index.get(&edge.source) else { continue };
            let Some(&tgt) = node_index.get(&edge.target) else { continue };
            if src != tgt {
                full_out_degrees[src] += 1;
                full_in_degrees[tgt] += 1;
            }
        }

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

            // Always record in typed_edges (for anomaly detection, modularity Q, etc.).
            typed_edges
                .entry(edge.kind.clone())
                .or_default()
                .push((src, tgt));

            // Defines edges encode containment hierarchy, not functional coupling.
            // Exclude them from the spectral adjacency matrix — they form a tree
            // that dominates eigenvectors and produces degenerate mega-modules.
            // Isolated nodes are later propagated via defines_parent_map().
            if edge.kind == "defines" {
                continue;
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
            edge_count += 1;
        }

        Graph {
            node_ids,
            node_index,
            adj,
            successors,
            predecessors,
            n,
            typed_edges,
            node_kinds,
            node_files,
            node_lines,
            edge_count,
            full_in_degrees,
            full_out_degrees,
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

    /// In-degree of a node (across all edge kinds in the graph).
    pub fn in_degree(&self, node: usize) -> usize {
        self.predecessors[node].len()
    }

    /// Out-degree of a node (across all edge kinds in the graph).
    pub fn out_degree(&self, node: usize) -> usize {
        self.successors[node].len()
    }

    /// Get edges of a specific kind.
    pub fn edges_of_kind(&self, kind: &str) -> &[(usize, usize)] {
        self.typed_edges.get(kind).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// Build child → parent map from "defines" edges.
    /// In a defines edge (src, tgt), the source defines (contains) the target.
    pub fn defines_parent_map(&self) -> HashMap<usize, usize> {
        let mut parent = HashMap::new();
        for &(src, tgt) in self.edges_of_kind("defines") {
            parent.insert(tgt, src);
        }
        parent
    }

    /// Build an anchor for a node from its stored metadata.
    pub fn anchor(&self, node: usize) -> crate::types::AnchorOutput {
        crate::types::AnchorOutput {
            node_id: self.node_ids[node].clone(),
            file: self.node_files[node].clone(),
            line: self.node_lines[node],
            kind: Some(self.node_kinds[node].clone()),
        }
    }
}
