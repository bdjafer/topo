use std::collections::{HashMap, HashSet};
use topo_analyzer::types::{AnalysisOutput, AnalyzerInput, EdgeEntry, ModuleOutput};

/// Deterministic xorshift64 RNG.
pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 1 } else { seed },
        }
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut s = self.state;
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        self.state = s;
        s
    }

    pub fn next_usize(&mut self, bound: usize) -> usize {
        if bound == 0 {
            return 0;
        }
        (self.next_u64() % bound as u64) as usize
    }

    /// Pick a random element from a slice.
    pub fn choice<'a, T>(&mut self, items: &'a [T]) -> &'a T {
        &items[self.next_usize(items.len())]
    }

    /// Sample k items without replacement (Fisher-Yates partial shuffle).
    pub fn sample<T: Clone>(&mut self, items: &[T], k: usize) -> Vec<T> {
        let mut pool: Vec<T> = items.to_vec();
        let k = k.min(pool.len());
        for i in 0..k {
            let j = i + self.next_usize(pool.len() - i);
            pool.swap(i, j);
        }
        pool.truncate(k);
        pool
    }
}

/// Build node_id -> module_id mapping from analysis output.
pub fn node_to_module(analysis: &AnalysisOutput) -> HashMap<String, usize> {
    let mut map = HashMap::new();
    for m in &analysis.architecture.modules {
        for nid in &m.members {
            map.insert(nid.clone(), m.id);
        }
    }
    map
}

/// Get non-unassigned modules with at least `min_size` members.
pub fn eligible_modules(analysis: &AnalysisOutput, min_size: usize) -> Vec<&ModuleOutput> {
    analysis
        .architecture
        .modules
        .iter()
        .filter(|m| !m.unassigned && m.size >= min_size)
        .collect()
}

/// Get node IDs already flagged by a specific issue kind.
pub fn flagged_nodes(analysis: &AnalysisOutput, kind: &str) -> HashSet<String> {
    let mut set = HashSet::new();
    for issue in &analysis.issues {
        if issue.kind == kind {
            for anchor in &issue.anchors {
                set.insert(anchor.node_id.clone());
            }
        }
    }
    set
}

/// Build in-degree map from roles.
pub fn in_degree_map(analysis: &AnalysisOutput) -> HashMap<String, usize> {
    analysis
        .roles
        .iter()
        .map(|r| (r.node_id.clone(), r.in_degree))
        .collect()
}

/// Filter node IDs to exclude test/spec nodes (avoids SCC suppression).
pub fn non_test_nodes(node_ids: &[String]) -> Vec<String> {
    node_ids
        .iter()
        .filter(|id| {
            let lower = id.to_lowercase();
            !lower.contains("test") && !lower.contains("spec") && !lower.contains("mock")
        })
        .cloned()
        .collect()
}

/// Check if a node name matches the overloaded_utility FP suppression list.
pub fn is_suppressed_utility_name(node_id: &str) -> bool {
    let last = node_id.rsplit('.').next().unwrap_or(node_id).to_lowercase();
    matches!(
        last.as_str(),
        "log" | "trace" | "debug" | "info" | "warn" | "error" | "println" | "print"
            | "serialize" | "deserialize" | "to_json" | "from_json" | "encode" | "decode"
            | "new" | "create" | "build" | "from" | "default" | "init"
    )
}

/// Deep-clone an AnalyzerInput, ready for mutation.
pub fn clone_input(input: &AnalyzerInput) -> AnalyzerInput {
    input.clone()
}

/// Add edges to a cloned input.
pub fn add_edges(input: &mut AnalyzerInput, edges: &[EdgeEntry]) {
    input.edges.extend(edges.iter().cloned());
}

/// Remove edges from a cloned input (match source + target + kind).
pub fn remove_edges(input: &mut AnalyzerInput, to_remove: &[EdgeEntry]) {
    let remove_set: HashSet<(&str, &str, &str)> = to_remove
        .iter()
        .map(|e| (e.source.as_str(), e.target.as_str(), e.kind.as_str()))
        .collect();
    input
        .edges
        .retain(|e| !remove_set.contains(&(e.source.as_str(), e.target.as_str(), e.kind.as_str())));
}

/// Build adjacency list (successors) for a subset of nodes from the input edges.
pub fn subgraph_adjacency(
    input: &AnalyzerInput,
    node_set: &HashSet<String>,
) -> HashMap<String, Vec<String>> {
    let mut adj: HashMap<String, Vec<String>> = HashMap::new();
    for nid in node_set {
        adj.entry(nid.clone()).or_default();
    }
    for edge in &input.edges {
        if edge.kind == "defines" {
            continue;
        }
        if node_set.contains(&edge.source) && node_set.contains(&edge.target) {
            adj.entry(edge.source.clone())
                .or_default()
                .push(edge.target.clone());
        }
    }
    adj
}

/// Count edges between two sets of nodes (excluding "defines").
pub fn count_edges_between(
    input: &AnalyzerInput,
    from: &HashSet<String>,
    to: &HashSet<String>,
) -> usize {
    input
        .edges
        .iter()
        .filter(|e| e.kind != "defines" && from.contains(&e.source) && to.contains(&e.target))
        .count()
}
