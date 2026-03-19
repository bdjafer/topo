//! Shared types for the topo-core API.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A node in the code graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeEntry {
    pub id: String,
    pub kind: String,
}

/// An edge in the code graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeEntry {
    pub source: String,
    pub target: String,
    pub kind: String,
}

/// Input to the analyzer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyzerInput {
    pub nodes: Vec<NodeEntry>,
    pub edges: Vec<EdgeEntry>,
    /// Number of clusters (auto-detect if None).
    #[serde(default)]
    pub k: Option<usize>,
    /// Which edge kinds to include (e.g. ["calls", "imports"]).
    #[serde(default)]
    pub edge_kinds: Option<Vec<String>>,
    /// Per-layer weights for multilayer analysis.
    #[serde(default)]
    pub layer_weights: Option<HashMap<String, f64>>,
}

/// Spectral decomposition result for a single connected component.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentResult {
    pub node_ids: Vec<String>,
    pub eigenvalues: Vec<f64>,
    /// Row-major eigenvector matrix: node_ids.len() rows × k columns.
    pub eigenvectors: Vec<Vec<f64>>,
}

/// Full output of the analyzer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyzerOutput {
    /// Spectral fingerprints: node_id → eigenvector coordinates.
    pub fingerprints: HashMap<String, Vec<f64>>,
    /// Cluster assignments: node_id → cluster_id.
    pub clusters: HashMap<String, usize>,
    /// Eigenvalues from the primary component.
    pub eigenvalues: Vec<f64>,
    /// Fiedler value (second-smallest Laplacian eigenvalue).
    pub fiedler_value: f64,
    /// Silhouette score of the clustering.
    pub silhouette: f64,
    /// Sizes of connected components (descending).
    pub component_sizes: Vec<usize>,
    /// Betweenness centrality: node_id → score.
    pub betweenness: HashMap<String, f64>,
    /// Strongly connected components with >1 node.
    pub sccs: Vec<Vec<String>>,
    /// Connected components.
    pub connected_components: Vec<Vec<String>>,
    /// Whether clustering was degenerate (fell back to package grouping).
    pub degenerate: bool,
}
