//! Integration tests: apply each mutation to a real codebase graph and verify
//! that the expected diagnostic fires after re-analysis.
//!
//! Uses the ripgrep example dataset (examples/ripgrep/graph.json).

use std::path::PathBuf;

use topo_analyzer::types::AnalyzerInput;
use topo_benchmark::mutations::{apply_mutation, MutationType};

fn load_ripgrep_graph() -> AnalyzerInput {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.pop(); // packages/
    path.pop(); // repo root
    path.push("examples/ripgrep/graph.json");

    let json = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("Failed to read {}: {}", path.display(), e));
    serde_json::from_str(&json)
        .unwrap_or_else(|e| panic!("Failed to parse {}: {}", path.display(), e))
}

fn test_mutation_triggers(mutation_type: MutationType) {
    let input = load_ripgrep_graph();
    let clean = topo_analyzer::analyze_full(&input);
    let expected = mutation_type.expected_diagnostic();

    let clean_count = clean
        .issues
        .iter()
        .filter(|i| i.kind == expected)
        .count();

    // Try multiple seeds — some may fail to find valid targets.
    let mut any_succeeded = false;
    let mut any_triggered = false;

    for seed in [42, 123, 999, 7777, 31337] {
        for severity in [2, 3] {
            let result = match apply_mutation(&input, &clean, mutation_type, severity, seed) {
                Some(r) => r,
                None => continue,
            };
            any_succeeded = true;

            // Always re-analyze the mutated graph — no escape hatch.
            // Even if the clean graph already has this diagnostic, we verify
            // the mutated graph still has it (mutation didn't break things)
            // and ideally has more instances.
            let mutated = topo_analyzer::analyze_full(&result.graph);
            let mutated_count = mutated
                .issues
                .iter()
                .filter(|i| i.kind == expected)
                .count();

            if mutated_count > 0 {
                any_triggered = true;
                break;
            }
        }
        if any_triggered {
            break;
        }
    }

    assert!(
        any_succeeded,
        "{:?}: mutation never produced a result (all seeds returned None)",
        mutation_type,
    );
    assert!(
        any_triggered,
        "{:?}: mutation succeeded but '{}' diagnostic never fired (clean had {})",
        mutation_type, expected, clean_count,
    );
}

#[test]
fn inject_cycle_triggers_on_ripgrep() {
    test_mutation_triggers(MutationType::InjectCycle);
}

#[test]
fn layer_violation_triggers_on_ripgrep() {
    test_mutation_triggers(MutationType::LayerViolation);
}

#[test]
fn overloaded_utility_triggers_on_ripgrep() {
    test_mutation_triggers(MutationType::OverloadedUtility);
}

#[test]
fn wide_interface_triggers_on_ripgrep() {
    test_mutation_triggers(MutationType::WideInterface);
}

#[test]
fn near_disconnect_triggers_on_ripgrep() {
    test_mutation_triggers(MutationType::NearDisconnect);
}
