//! Rust source code parser for topo.
//!
//! Parses Rust codebases into a typed multilayer graph matching
//! `graph.schema.json`. Uses rust-analyzer's semantic analysis engine
//! for full type inference, name resolution, and call graph extraction.

pub mod call_extract;
pub mod graph;
pub mod hir_walk;
pub mod source_map;
pub mod workspace;

use std::collections::{HashMap, HashSet};
use std::path::Path;

use anyhow::Result;
use ra_ap_hir::{Semantics, attach_db};

use graph::CodeGraph;
use hir_walk::HirEntity;

/// Parse a Rust project and return a CodeGraph JSON string.
///
/// If `root` is a workspace, all member crates are parsed into a single graph.
/// If `root` is a single crate, only that crate is parsed.
pub fn parse_project(root: &Path) -> Result<String> {
    let graph = parse_project_to_graph(root)?;
    graph.to_json()
}

/// Parse a Rust project and return a CodeGraph struct.
pub fn parse_project_to_graph(root: &Path) -> Result<CodeGraph> {
    let root = dunce::canonicalize(root).unwrap_or_else(|_| root.to_path_buf());

    // 1. Load the workspace via rust-analyzer.
    let (db, vfs) = workspace::load_project(&root)?;

    // 2. Attach the hir_ty database to this thread (required by ra_ap_hir_ty).
    attach_db(&db, || extract_graph(&db, &vfs, &root))
}

/// Core extraction logic, runs inside `db.attach()`.
fn extract_graph(
    db: &ra_ap_ide_db::RootDatabase,
    vfs: &ra_ap_vfs::Vfs,
    root: &Path,
) -> Result<CodeGraph> {
    let sema = Semantics::new(db);
    let mapper = source_map::SourceMapper::new(root);

    // Find workspace-member crates.
    let crates = workspace::workspace_crates(db, vfs, root);
    if crates.is_empty() {
        anyhow::bail!("No workspace-member crates found in {}", root.display());
    }

    let mut graph = CodeGraph::new();
    let mut entity_map: HashMap<HirEntity, String> = HashMap::new();

    // Pass 1: Extract all nodes + CONTAINS edges across all crates.
    // This populates entity_map with every entity so that cross-crate
    // lookups succeed in pass 2.
    for krate in &crates {
        hir_walk::extract_nodes(db, vfs, &mapper, krate, &mut graph, &mut entity_map);
    }

    // Pass 2: Extract relationship edges (IMPORTS, INHERITS, CALLS).
    // entity_map now contains all crates, so cross-crate edges resolve.
    for krate in &crates {
        hir_walk::extract_imports(db, krate, &mut graph, &entity_map);
        hir_walk::extract_inherits(db, krate, &mut graph, &entity_map);
        call_extract::extract_calls(db, &sema, krate, &mut graph, &entity_map);
    }

    dedup_edges(&mut graph);
    graph.sort();

    Ok(graph)
}

/// Remove duplicate edges (same source, target, kind).
fn dedup_edges(graph: &mut CodeGraph) {
    let mut seen = HashSet::new();
    graph.edges.retain(|e| {
        let key = (e.source.clone(), e.target.clone(), e.kind.to_string());
        seen.insert(key)
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn topo_analyzer_path() -> PathBuf {
        let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        manifest.parent().unwrap().join("topo-analyzer")
    }

    #[test]
    fn test_parse_topo_analyzer_produces_valid_graph() {
        let path = topo_analyzer_path();
        if !path.exists() {
            return;
        }
        let graph = parse_project_to_graph(&path).unwrap();

        let modules = graph.nodes.iter().filter(|n| n.kind == "module").count();
        let classes = graph.nodes.iter().filter(|n| n.kind == "class").count();
        let functions = graph.nodes.iter().filter(|n| n.kind == "function").count();

        assert!(modules >= 10, "Expected >=10 modules, got {modules}");
        assert!(classes >= 5, "Expected >=5 classes, got {classes}");
        assert!(functions >= 20, "Expected >=20 functions, got {functions}");

        let defines = graph.edges.iter().filter(|e| e.kind == "defines").count();
        let imports = graph.edges.iter().filter(|e| e.kind == "imports").count();
        let calls = graph.edges.iter().filter(|e| e.kind == "calls").count();

        assert!(defines >= 30, "Expected >=30 defines edges, got {defines}");
        assert!(imports >= 5, "Expected >=5 import edges, got {imports}");
        assert!(calls >= 10, "Expected >=10 call edges, got {calls}");

        let json = graph.to_json().unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(parsed["nodes"].is_array());
        assert!(parsed["edges"].is_array());
    }

    #[test]
    fn bench_parse_topo_analyzer() {
        let path = topo_analyzer_path();
        if !path.exists() {
            return;
        }

        const RUNS: usize = 5;
        let mut timings = Vec::with_capacity(RUNS);
        let mut last_nodes = 0;
        let mut last_edges = 0;

        for i in 0..RUNS {
            let start = std::time::Instant::now();
            let graph = parse_project_to_graph(&path).unwrap();
            let elapsed = start.elapsed();
            timings.push(elapsed);

            let nodes = graph.nodes.len();
            let edges = graph.edges.len();
            let calls = graph.edges.iter().filter(|e| e.kind == "calls").count();
            let imports = graph.edges.iter().filter(|e| e.kind == "imports").count();
            let defines = graph.edges.iter().filter(|e| e.kind == "defines").count();
            let inherits = graph.edges.iter().filter(|e| e.kind == "inherits").count();

            eprintln!(
                "  run {}: {:>6.0?}  ({nodes} nodes, {edges} edges: {calls} calls, {imports} imports, {defines} defines, {inherits} inherits)",
                i + 1, elapsed
            );

            if i > 0 {
                assert_eq!(nodes, last_nodes, "Node count changed between runs");
                assert_eq!(edges, last_edges, "Edge count changed between runs");
            }
            last_nodes = nodes;
            last_edges = edges;
        }

        timings.sort();
        let min = timings[0];
        let median = timings[RUNS / 2];
        let max = timings[RUNS - 1];
        eprintln!("\n  min={min:>6.0?}  median={median:>6.0?}  max={max:>6.0?}");
    }

    #[test]
    fn test_parse_topo_analyzer_json_schema_conformance() {
        let path = topo_analyzer_path();
        if !path.exists() {
            return;
        }
        let json = parse_project(&path).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();

        for node in parsed["nodes"].as_array().unwrap() {
            assert!(node["id"].is_string(), "Node missing id: {node}");
            let kind = node["kind"].as_str().unwrap();
            assert!(
                ["module", "class", "function"].contains(&kind),
                "Invalid kind: {kind}"
            );
        }

        for edge in parsed["edges"].as_array().unwrap() {
            assert!(edge["source"].is_string(), "Edge missing source: {edge}");
            assert!(edge["target"].is_string(), "Edge missing target: {edge}");
            let kind = edge["kind"].as_str().unwrap();
            assert!(
                ["calls", "imports", "inherits", "defines"].contains(&kind),
                "Invalid edge kind: {kind}"
            );
        }
    }
}
