//! topo-parser-rs: Rust source code parser for topo.
//!
//! Parses Rust codebases into a typed multilayer graph matching
//! `graph.schema.json`. Produces the same JSON format as the Python
//! parser (`topo-parser`), enabling the analyzer to process Rust projects.

pub mod calls;
pub mod discovery;
pub mod graph;
pub mod imports;
pub mod inherits;
pub mod items;
pub mod module_tree;

use std::collections::HashSet;
use std::path::Path;

use anyhow::{Context, Result};

use graph::{CodeGraph, Node};

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
    let crates = discovery::discover_crates(root)?;
    if crates.is_empty() {
        anyhow::bail!("No Rust crates found in {}", root.display());
    }

    let mut graph = CodeGraph::new();
    let root_canonical = root.canonicalize().unwrap_or_else(|_| root.to_path_buf());

    for krate in &crates {
        parse_crate(&mut graph, krate, &root_canonical)?;
    }

    // Deduplicate import edges (same source→target pair).
    dedup_edges(&mut graph);
    graph.sort();

    Ok(graph)
}

/// Parse a single crate into the graph.
fn parse_crate(graph: &mut CodeGraph, krate: &discovery::CrateInfo, project_root: &Path) -> Result<()> {
    // 1. Resolve module tree: file → module ID.
    let file_to_module = module_tree::resolve_module_tree(krate)
        .with_context(|| format!("Failed to resolve module tree for {}", krate.name))?;

    // 2. Create module nodes.
    for (file, module_id) in &file_to_module {
        let relative_file = make_relative(file, project_root);
        let name = module_id.rsplit('.').next().unwrap_or(module_id).to_string();
        graph.add_node(Node {
            id: module_id.clone(),
            kind: "module",
            file: Some(relative_file),
            line: Some(1),
            name: Some(name),
        });
    }

    // Add CONTAINS edges from parent modules to child modules.
    let module_ids: Vec<String> = file_to_module.values().cloned().collect();
    for module_id in &module_ids {
        if let Some((parent, _)) = module_id.rsplit_once('.') {
            if module_ids.iter().any(|m| m == parent) {
                graph.add_edge(graph::Edge {
                    source: parent.to_string(),
                    target: module_id.clone(),
                    kind: "contains",
                });
            }
        }
    }

    // 3. Extract items (functions, structs, etc.) from each file.
    for (file, module_id) in &file_to_module {
        let relative_file = make_relative(file, project_root);
        if let Err(e) = items::extract_items(graph, file, module_id, &relative_file) {
            eprintln!("Warning: failed to extract items from {}: {e}", file.display());
        }
    }

    // 4. Extract imports.
    for (file, module_id) in &file_to_module {
        if let Err(e) = imports::extract_imports(graph, file, module_id, &krate.name) {
            eprintln!("Warning: failed to extract imports from {}: {e}", file.display());
        }
    }

    // 5. Extract calls.
    for (file, module_id) in &file_to_module {
        let import_map = items::collect_import_map(file, module_id, &krate.name)
            .unwrap_or_default();
        if let Err(e) = calls::extract_calls(graph, file, module_id, &krate.name, &import_map) {
            eprintln!("Warning: failed to extract calls from {}: {e}", file.display());
        }
    }

    // 6. Extract inherits.
    for (file, module_id) in &file_to_module {
        if let Err(e) = inherits::extract_inherits(graph, file, module_id) {
            eprintln!("Warning: failed to extract inherits from {}: {e}", file.display());
        }
    }

    Ok(())
}

/// Make a path relative to the project root.
fn make_relative(file: &Path, project_root: &Path) -> String {
    file.strip_prefix(project_root)
        .unwrap_or(file)
        .to_string_lossy()
        .to_string()
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
        // Find the topo-analyzer crate relative to this crate.
        let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        manifest.parent().unwrap().join("topo-analyzer")
    }

    #[test]
    fn test_discover_topo_analyzer() {
        let path = topo_analyzer_path();
        if !path.exists() {
            return; // Skip if not in the workspace.
        }
        let crates = discovery::discover_crates(&path).unwrap();
        assert_eq!(crates.len(), 1);
        assert_eq!(crates[0].name, "topo_analyzer");
        assert!(crates[0].src_files.len() >= 10);
    }

    #[test]
    fn test_module_tree_topo_analyzer() {
        let path = topo_analyzer_path();
        if !path.exists() {
            return;
        }
        let crates = discovery::discover_crates(&path).unwrap();
        let file_to_module = module_tree::resolve_module_tree(&crates[0]).unwrap();

        // Should have at least 10 modules.
        assert!(file_to_module.len() >= 10, "Got {} modules", file_to_module.len());

        // Check specific module IDs.
        let module_ids: Vec<&str> = file_to_module.values().map(|s| s.as_str()).collect();
        assert!(module_ids.contains(&"topo_analyzer"), "Missing root module");
        assert!(module_ids.contains(&"topo_analyzer.graph"), "Missing graph module");
        assert!(module_ids.contains(&"topo_analyzer.types"), "Missing types module");
        assert!(module_ids.contains(&"topo_analyzer.spectral"), "Missing spectral module");
        assert!(module_ids.contains(&"topo_analyzer.roles"), "Missing roles module");
    }

    #[test]
    fn test_parse_topo_analyzer_produces_valid_graph() {
        let path = topo_analyzer_path();
        if !path.exists() {
            return;
        }
        let graph = parse_project_to_graph(&path).unwrap();

        // Count by kind.
        let modules = graph.nodes.iter().filter(|n| n.kind == "module").count();
        let classes = graph.nodes.iter().filter(|n| n.kind == "class").count();
        let functions = graph.nodes.iter().filter(|n| n.kind == "function").count();

        assert!(modules >= 10, "Expected >=10 modules, got {modules}");
        assert!(classes >= 5, "Expected >=5 classes, got {classes}");
        assert!(functions >= 20, "Expected >=20 functions, got {functions}");

        // Count edges by kind.
        let contains = graph.edges.iter().filter(|e| e.kind == "contains").count();
        let imports = graph.edges.iter().filter(|e| e.kind == "imports").count();
        let calls = graph.edges.iter().filter(|e| e.kind == "calls").count();

        assert!(contains >= 30, "Expected >=30 contains edges, got {contains}");
        assert!(imports >= 5, "Expected >=5 import edges, got {imports}");
        // Calls may be low due to conservative resolution — just check non-zero.
        assert!(calls >= 1, "Expected >=1 call edges, got {calls}");

        // Verify JSON serializes.
        let json = graph.to_json().unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(parsed["nodes"].is_array());
        assert!(parsed["edges"].is_array());
    }

    #[test]
    fn test_parse_topo_analyzer_json_schema_conformance() {
        let path = topo_analyzer_path();
        if !path.exists() {
            return;
        }
        let json = parse_project(&path).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();

        // Every node must have "id" and "kind".
        for node in parsed["nodes"].as_array().unwrap() {
            assert!(node["id"].is_string(), "Node missing id: {node}");
            let kind = node["kind"].as_str().unwrap();
            assert!(
                ["module", "class", "function"].contains(&kind),
                "Invalid kind: {kind}"
            );
        }

        // Every edge must have "source", "target", "kind".
        for edge in parsed["edges"].as_array().unwrap() {
            assert!(edge["source"].is_string(), "Edge missing source: {edge}");
            assert!(edge["target"].is_string(), "Edge missing target: {edge}");
            let kind = edge["kind"].as_str().unwrap();
            assert!(
                ["calls", "imports", "inherits", "contains"].contains(&kind),
                "Invalid edge kind: {kind}"
            );
        }
    }
}
