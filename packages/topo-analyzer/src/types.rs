//! Shared types for the topo-core API.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Node type vocabulary for R-GIN embedding lookup
// ---------------------------------------------------------------------------

/// Canonical node type vocabulary for R-GIN embedding lookup.
/// DO NOT reorder — indices are baked into trained model weights.
pub const NODE_TYPE_VOCAB: &[&str] = &[
    "function",  // 0 — behavioral unit, coupling endpoint
    "module",    // 1 — container, import hub
    "class",     // 2 — concrete type (struct, enum, class, dataclass)
    "interface", // 3 — abstract contract (trait, interface, protocol)
];

/// Fallback index for unknown node kinds.
pub const UNKNOWN_TYPE_INDEX: usize = NODE_TYPE_VOCAB.len();

/// Map a node kind string to the canonical vocabulary index.
///
/// Returns the index into `NODE_TYPE_VOCAB`, or `UNKNOWN_TYPE_INDEX` for
/// unrecognized kinds.
pub fn node_type_index(kind: &str) -> usize {
    // Exact match first
    if let Some(i) = NODE_TYPE_VOCAB.iter().position(|&v| v == kind) {
        return i;
    }
    // Case-insensitive fallback
    let lower = kind.to_ascii_lowercase();
    if let Some(i) = NODE_TYPE_VOCAB.iter().position(|&v| v == lower) {
        return i;
    }
    UNKNOWN_TYPE_INDEX
}

#[cfg(test)]
mod node_type_tests {
    use super::*;

    #[test]
    fn test_exact_matches() {
        assert_eq!(node_type_index("function"), 0);
        assert_eq!(node_type_index("module"), 1);
        assert_eq!(node_type_index("class"), 2);
        assert_eq!(node_type_index("interface"), 3);
    }

    #[test]
    fn test_case_insensitive() {
        assert_eq!(node_type_index("Function"), 0);
        assert_eq!(node_type_index("MODULE"), 1);
        assert_eq!(node_type_index("Class"), 2);
        assert_eq!(node_type_index("INTERFACE"), 3);
    }

    #[test]
    fn test_unknown_fallback() {
        assert_eq!(node_type_index("widget"), UNKNOWN_TYPE_INDEX);
        assert_eq!(node_type_index(""), UNKNOWN_TYPE_INDEX);
        assert_eq!(node_type_index("impl"), UNKNOWN_TYPE_INDEX);
    }

    #[test]
    fn test_vocab_size() {
        assert_eq!(NODE_TYPE_VOCAB.len(), 4);
        assert_eq!(UNKNOWN_TYPE_INDEX, 4);
    }
}

// ---------------------------------------------------------------------------
// Input types
// ---------------------------------------------------------------------------

/// A node in the code graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeEntry {
    pub id: String,
    pub kind: String,
    #[serde(default)]
    pub file: Option<String>,
    #[serde(default)]
    pub line: Option<u32>,
    #[serde(default)]
    pub line_end: Option<u32>,
}

/// An edge in the code graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeEntry {
    pub source: String,
    pub target: String,
    pub kind: String,
}

/// Scope/projection metadata passed through to the output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScopeInput {
    pub level: String,
    pub edge_kinds: Vec<String>,
    #[serde(default)]
    pub internal_only: bool,
    #[serde(default)]
    pub roots: Vec<String>,
}

/// Projection configuration — when present, analyze_full projects the raw
/// graph before analysis (filter by kind/scope, lift IDs, remap edges).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectionInput {
    /// Target analysis level: "symbol", "module", or "package".
    pub level: String,
    /// Node kinds to include (e.g. ["module", "class", "function"]).
    pub source_node_kinds: Vec<String>,
    /// Edge kinds to include (e.g. ["calls", "imports", "inherits"]).
    pub edge_kinds: Vec<String>,
    /// Scope roots — file paths must be under one of these prefixes.
    /// Empty means no scope filtering.
    #[serde(default)]
    pub scope_roots: Vec<String>,
    /// Only keep edges where both endpoints are in scope.
    #[serde(default = "default_true")]
    pub internal_only: bool,
}

fn default_true() -> bool {
    true
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
    /// Scope/projection metadata (pass-through to output).
    #[serde(default)]
    pub scope: Option<ScopeInput>,
    /// Raw parsed graph node count (before projection), for coverage.
    #[serde(default)]
    pub parsed_nodes: Option<usize>,
    /// Raw parsed graph edge count (before projection), for coverage.
    #[serde(default)]
    pub parsed_edges: Option<usize>,
    /// Self-edge ratio from projection (edges collapsed into self-edges).
    #[serde(default)]
    pub self_edge_ratio: Option<f64>,
    /// When present, the input is a raw (unprojected) graph and Rust
    /// performs the projection before analysis.
    #[serde(default)]
    pub projection: Option<ProjectionInput>,
    /// Package/crate names for top-level architecture grouping.
    /// When set with 2+ packages, used instead of spectral clustering for
    /// module assignment. The parser populates this from workspace structure.
    #[serde(default)]
    pub packages: Option<Vec<String>>,
    /// Pre-computed semantic embeddings: node_id -> 768-dim vector.
    /// Optional — when present, enables semantic analysis tools.
    #[serde(default)]
    pub semantic_embeddings: Option<HashMap<String, Vec<f32>>>,
    /// Enable experimental diagnostics (shadow-dependency). O(n²) cost.
    #[serde(default)]
    pub experimental: Option<bool>,
}

// ---------------------------------------------------------------------------
// Legacy output (kept as CoreOutput for backward compatibility)
// ---------------------------------------------------------------------------

/// Spectral decomposition result for a single connected component.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentResult {
    pub node_ids: Vec<String>,
    pub eigenvalues: Vec<f64>,
    /// Row-major eigenvector matrix: node_ids.len() rows × k columns.
    pub eigenvectors: Vec<Vec<f64>>,
}

/// Legacy output of the core analyzer (spectral + clustering + graph algos).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyzerOutput {
    pub fingerprints: HashMap<String, Vec<f64>>,
    pub clusters: HashMap<String, usize>,
    pub eigenvalues: Vec<f64>,
    pub fiedler_value: f64,
    pub silhouette: f64,
    pub component_sizes: Vec<usize>,
    pub betweenness: HashMap<String, f64>,
    pub sccs: Vec<Vec<String>>,
    pub connected_components: Vec<Vec<String>>,
    pub degenerate: bool,
}

// ---------------------------------------------------------------------------
// Schema-compliant output (matches analysis.schema.json)
// ---------------------------------------------------------------------------

/// Source code location reference.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnchorOutput {
    pub node_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub line: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kind: Option<String>,
}

/// Scope metadata in the output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScopeOutput {
    pub level: String,
    pub edge_kinds: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub internal_only: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub roots: Option<Vec<String>>,
}

/// Coverage: how much of the codebase was analyzed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoverageOutput {
    pub analyzed_nodes: usize,
    pub analyzed_edges: usize,
    pub parsed_nodes: usize,
    pub parsed_edges: usize,
}

/// Spectral decomposition summary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectralOutput {
    pub fiedler_value: f64,
    pub eigenvalues: Vec<f64>,
    pub nodes_covered: usize,
    pub coverage_ratio: f64,
    pub components: usize,
    pub largest_component_ratio: f64,
}

/// A detected structural module.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleOutput {
    pub id: usize,
    pub label: String,
    pub size: usize,
    pub members: Vec<String>,
    pub cohesion: Option<f64>,
    pub separation: Option<f64>,
    pub confidence: f64,
    pub unassigned: bool,
    /// Nodes assigned via defines-tree propagation (not spectral clustering).
    #[serde(skip_serializing_if = "is_zero")]
    pub propagated_count: usize,
    /// Top TF-IDF terms from node ID tokenization. Available without semantic embeddings.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_terms: Option<Vec<String>>,
    /// Average pairwise cosine similarity of semantic embeddings within this module.
    /// Null if module size < 6 or semantic analysis not enabled.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub semantic_coherence: Option<f64>,
}

fn is_zero(v: &usize) -> bool {
    *v == 0
}

/// Directed dependency between two modules.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencyOutput {
    pub source: usize,
    pub target: usize,
    pub weight: usize,
    pub edge_kinds: HashMap<String, usize>,
}

/// Architecture section: modules + inter-module dependencies.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchitectureOutput {
    pub modules: Vec<ModuleOutput>,
    pub dependencies: Vec<DependencyOutput>,
    pub silhouette: Option<f64>,
    pub package_fallback: bool,
    /// Comparison of spectral modules against declared package boundaries.
    /// Present only when the input has 2+ packages.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub package_agreement: Option<PackageAgreementOutput>,
}

/// Comparison of spectral modules against declared package boundaries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackageAgreementOutput {
    /// NMI between spectral and package partitions. 1.0 = perfect alignment.
    pub nmi: f64,
    /// Per-module breakdown of which packages contributed members.
    pub module_composition: Vec<ModuleCompositionOutput>,
}

/// Per-module breakdown of package membership.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleCompositionOutput {
    pub module_id: usize,
    /// Package name -> count of members from that package.
    pub packages: HashMap<String, usize>,
    /// True if this module draws members from 2+ packages.
    pub cross_package: bool,
}

/// Structural role of a single node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoleOutput {
    pub node_id: String,
    pub role: String,
    pub degree: usize,
    pub betweenness: f64,
    pub in_degree: usize,
    pub out_degree: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub anchor: Option<AnchorOutput>,
    /// Semantic local variation: how much this node disagrees with its structural neighbors.
    /// Only present when semantic analysis is enabled.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub local_variation: Option<f64>,
}

/// A prioritized structural issue.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct IssueOutput {
    pub id: String,
    pub kind: String,
    pub title: String,
    pub description: String,
    pub severity: f64,
    pub severity_label: String,
    pub confidence: f64,
    pub confidence_label: String,
    pub anchors: Vec<AnchorOutput>,
    /// For misplaced_concern: the module the node should move to.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub suggested_module: Option<String>,
    /// For misplaced_concern: cosine similarity to own module centroid.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub similarity_own: Option<f64>,
    /// For misplaced_concern: cosine similarity to best other module centroid.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub similarity_best: Option<f64>,
    /// For cross_package_coupling: classified root cause.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub root_cause: Option<String>,
    /// For cross_package_coupling: semantic coherence of the module.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub semantic_coherence: Option<f64>,
}

/// Semantic energy profile: how disagreement distributes across structural scales.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticEnergyProfile {
    pub eigenvalues: Vec<f64>,
    pub semantic_energy: Vec<f64>,
}

/// Structural health metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthOutput {
    pub modularity_q: Option<f64>,
    /// Fraction of nodes that received spectral fingerprints.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub spectral_coverage_ratio: Option<f64>,
    /// Fraction of scoped edges that collapsed into self-edges at the current level.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub self_edge_drop_ratio: Option<f64>,
    /// Rayleigh quotient: how smoothly semantics vary over the graph. Lower = better organized.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub semantic_smoothness: Option<f64>,
    /// AMI between structural clusters and semantic clusters. Higher = better alignment.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub semantic_structural_ami: Option<f64>,
    /// GFT energy profile: semantic energy at each structural scale.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub semantic_energy_profile: Option<SemanticEnergyProfile>,
}

/// Complete analysis output matching analysis.schema.json.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisOutput {
    pub scope: ScopeOutput,
    pub coverage: CoverageOutput,
    pub spectral: Option<SpectralOutput>,
    pub architecture: ArchitectureOutput,
    pub roles: Vec<RoleOutput>,
    pub issues: Vec<IssueOutput>,
    pub health: Option<HealthOutput>,
    /// Whether semantic analysis was attempted and the quality gate passed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub semantic_enabled: Option<bool>,
}
