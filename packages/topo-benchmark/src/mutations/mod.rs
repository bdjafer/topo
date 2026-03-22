//! Deterministic mutation operators for structural benchmark evaluation.
//!
//! Each operator takes an `AnalyzerInput` graph, the clean `AnalysisOutput`,
//! a severity level (1/2/3), and a seed. It returns a `MutationResult`
//! containing the mutated graph plus metadata about what changed.
//!
//! The operators use the clean analysis to make *smart* targeting decisions:
//! picking structurally significant positions, avoiding already-flagged regions,
//! and calibrating mutation magnitude to the graph's existing structure.

pub mod types;
pub mod helpers;
pub mod inject_cycle;
pub mod layer_violation;
pub mod overloaded_utility;
pub mod wide_interface;
pub mod near_disconnect;

pub use types::{MutationResult, MutationType};

use topo_analyzer::types::{AnalyzerInput, AnalysisOutput};

/// Apply a named mutation to a graph.
///
/// Returns `None` if the graph lacks preconditions for the mutation
/// (too few modules, too few nodes, etc.).
pub fn apply_mutation(
    input: &AnalyzerInput,
    analysis: &AnalysisOutput,
    mutation_type: MutationType,
    severity: u8,
    seed: u64,
) -> Option<MutationResult> {
    match mutation_type {
        MutationType::InjectCycle => inject_cycle::mutate(input, analysis, severity, seed),
        MutationType::LayerViolation => layer_violation::mutate(input, analysis, severity, seed),
        MutationType::OverloadedUtility => overloaded_utility::mutate(input, analysis, severity, seed),
        MutationType::WideInterface => wide_interface::mutate(input, analysis, severity, seed),
        MutationType::NearDisconnect => near_disconnect::mutate(input, analysis, severity, seed),
    }
}
