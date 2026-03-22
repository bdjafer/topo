use serde::{Deserialize, Serialize};
use topo_analyzer::types::{AnalyzerInput, EdgeEntry};

/// Which mutation to apply.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MutationType {
    InjectCycle,
    LayerViolation,
    OverloadedUtility,
    WideInterface,
    NearDisconnect,
}

impl MutationType {
    /// The issue kind this mutation is expected to trigger.
    pub fn expected_diagnostic(&self) -> &'static str {
        match self {
            Self::InjectCycle => "circular_dependency",
            Self::LayerViolation => "layer_violation",
            Self::OverloadedUtility => "overloaded_utility",
            Self::WideInterface => "wide_interface",
            Self::NearDisconnect => "near_disconnect",
        }
    }

    pub fn all() -> &'static [MutationType] {
        &[
            Self::InjectCycle,
            Self::LayerViolation,
            Self::OverloadedUtility,
            Self::WideInterface,
            Self::NearDisconnect,
        ]
    }
}

/// Result of applying a mutation operator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MutationResult {
    /// The mutated graph (ready to pass to `analyze_full`).
    pub graph: AnalyzerInput,
    /// Which mutation was applied.
    pub mutation_type: MutationType,
    /// The issue kind the mutation should trigger.
    pub expected_diagnostic: String,
    /// Severity level (1 = mild, 2 = medium, 3 = severe).
    pub severity_level: u8,
    /// Seed used for deterministic targeting.
    pub seed: u64,
    /// Edges added by the mutation.
    pub added_edges: Vec<EdgeEntry>,
    /// Edges removed by the mutation.
    pub removed_edges: Vec<EdgeEntry>,
    /// Node IDs in the mutated region (for attribution@3 scoring).
    pub modified_region: Vec<String>,
    /// Human-readable description of the mutation.
    pub description: String,
}
