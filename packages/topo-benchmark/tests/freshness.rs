//! Freshness check: detects when the parser output diverges from frozen benchmark graphs.
//!
//! If this test fails, run:
//!   cargo run -p topo-cli -- parse packages/topo-analyzer/ -o benchmark/datasets/architecture/topo_analyzer/graph.json
//! Then update labels.json and derived datasets (mutations, stability, anomalies) accordingly.

use std::collections::BTreeSet;
use std::process::Command;

/// Parse topo-analyzer with the CLI and compare node IDs against the frozen graph.
///
/// This test is ignored by default because it requires the full parser toolchain
/// (ra_ap_* crates, which are slow to compile). Run explicitly with:
///   cargo test -p topo-benchmark --test freshness -- --ignored
#[test]
#[ignore]
fn frozen_graph_matches_current_parser() {
    let workspace_root = find_workspace_root();

    // Parse topo-analyzer using the CLI.
    let output = Command::new("cargo")
        .args([
            "run",
            "-p",
            "topo-cli",
            "--",
            "parse",
            "packages/topo-analyzer/",
        ])
        .current_dir(&workspace_root)
        .output()
        .expect("failed to run cargo run -p topo-cli -- parse");

    assert!(
        output.status.success(),
        "Parser failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let fresh: serde_json::Value =
        serde_json::from_slice(&output.stdout).expect("parser output is not valid JSON");

    // Load the frozen graph.
    let frozen_path = workspace_root
        .join("benchmark/datasets/architecture/topo_analyzer/graph.json");
    let frozen_text = std::fs::read_to_string(&frozen_path)
        .unwrap_or_else(|_| panic!("cannot read {}", frozen_path.display()));
    let frozen: serde_json::Value =
        serde_json::from_str(&frozen_text).expect("frozen graph is not valid JSON");

    // Compare node ID sets.
    let fresh_ids: BTreeSet<&str> = fresh["nodes"]
        .as_array()
        .expect("fresh nodes is not array")
        .iter()
        .map(|n| n["id"].as_str().expect("node missing id"))
        .collect();

    let frozen_ids: BTreeSet<&str> = frozen["nodes"]
        .as_array()
        .expect("frozen nodes is not array")
        .iter()
        .map(|n| n["id"].as_str().expect("node missing id"))
        .collect();

    let added: Vec<&&str> = fresh_ids.difference(&frozen_ids).collect();
    let removed: Vec<&&str> = frozen_ids.difference(&fresh_ids).collect();

    if !added.is_empty() || !removed.is_empty() {
        let mut msg = String::from(
            "Frozen benchmark graph is stale — parser output has diverged.\n\n",
        );
        if !added.is_empty() {
            msg.push_str(&format!("New nodes (not in frozen graph): {:?}\n", added));
        }
        if !removed.is_empty() {
            msg.push_str(&format!(
                "Missing nodes (in frozen graph but not in parser output): {:?}\n",
                removed
            ));
        }
        msg.push_str(
            "\nTo fix, regenerate the frozen graph:\n  \
             cargo run -p topo-cli -- parse packages/topo-analyzer/ \
             > benchmark/datasets/architecture/topo_analyzer/graph.json\n\
             Then update labels.json and derived datasets.",
        );
        panic!("{msg}");
    }

    // Also check edge count hasn't changed dramatically (>10% drift).
    let fresh_edge_count = fresh["edges"].as_array().map(|a| a.len()).unwrap_or(0);
    let frozen_edge_count = frozen["edges"].as_array().map(|a| a.len()).unwrap_or(0);
    let drift = (fresh_edge_count as f64 - frozen_edge_count as f64).abs()
        / frozen_edge_count.max(1) as f64;

    assert!(
        drift <= 0.10,
        "Edge count drifted by {:.0}% (frozen: {frozen_edge_count}, fresh: {fresh_edge_count}). \
         Regenerate the frozen graph.",
        drift * 100.0
    );
}

/// Also check that gold labels reference only nodes that exist in the frozen graph.
#[test]
fn gold_labels_reference_valid_nodes() {
    let workspace_root = find_workspace_root();
    let graph_path = workspace_root
        .join("benchmark/datasets/architecture/topo_analyzer/graph.json");

    if !graph_path.exists() {
        // Dataset not yet generated — skip silently.
        return;
    }

    let graph_text = std::fs::read_to_string(&graph_path).unwrap();
    let graph: serde_json::Value = serde_json::from_str(&graph_text).unwrap();
    let node_ids: BTreeSet<String> = graph["nodes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|n| n["id"].as_str().unwrap().to_string())
        .collect();

    let labels_path = workspace_root
        .join("benchmark/datasets/architecture/topo_analyzer/labels.json");
    if !labels_path.exists() {
        return;
    }
    let labels_text = std::fs::read_to_string(&labels_path).unwrap();
    let labels: serde_json::Value = serde_json::from_str(&labels_text).unwrap();

    let mut dangling = Vec::new();
    if let Some(included) = labels["included_nodes"].as_object() {
        for node_id in included.keys() {
            if !node_ids.contains(node_id.as_str()) {
                dangling.push(node_id.clone());
            }
        }
    }

    assert!(
        dangling.is_empty(),
        "Gold labels reference {} nodes not in the frozen graph: {:?}\n\
         Either update labels.json or regenerate graph.json.",
        dangling.len(),
        dangling
    );
}

/// Check that mutation case mutated_region nodes exist in their graph.
#[test]
fn mutation_regions_reference_valid_nodes() {
    let workspace_root = find_workspace_root();
    let mutations_dir = workspace_root.join("benchmark/datasets/mutations");

    if !mutations_dir.exists() {
        return;
    }

    for entry in std::fs::read_dir(&mutations_dir).unwrap() {
        let case_dir = entry.unwrap().path();
        if !case_dir.is_dir() {
            continue;
        }

        let expectations_path = case_dir.join("expectations.json");
        let clean_path = case_dir.join("variants/clean.json");
        if !expectations_path.exists() || !clean_path.exists() {
            continue;
        }

        let graph_text = std::fs::read_to_string(&clean_path).unwrap();
        let graph: serde_json::Value = serde_json::from_str(&graph_text).unwrap();
        let node_ids: BTreeSet<String> = graph["nodes"]
            .as_array()
            .unwrap()
            .iter()
            .map(|n| n["id"].as_str().unwrap().to_string())
            .collect();

        let exp_text = std::fs::read_to_string(&expectations_path).unwrap();
        let exp: serde_json::Value = serde_json::from_str(&exp_text).unwrap();

        if let Some(region) = exp.get("mutated_region") {
            if let Some(nodes) = region["nodes"].as_array() {
                let case_name = case_dir.file_name().unwrap().to_string_lossy();
                for node in nodes {
                    let nid = node.as_str().unwrap();
                    // Check in clean graph — some mutation regions reference nodes
                    // that only exist in the mutated variant (e.g., bridge.connector).
                    // For topo_* cases, all nodes should exist in clean.
                    if case_name.starts_with("topo_") && !node_ids.contains(nid) {
                        panic!(
                            "Mutation case {}: mutated_region node '{}' not in clean.json",
                            case_name, nid
                        );
                    }
                }
            }
        }
    }
}

fn find_workspace_root() -> std::path::PathBuf {
    let mut dir = std::env::current_dir().unwrap();
    loop {
        if dir.join("Cargo.toml").exists() && dir.join("benchmark").exists() {
            return dir;
        }
        if !dir.pop() {
            panic!("Could not find workspace root (directory containing Cargo.toml and benchmark/)");
        }
    }
}
