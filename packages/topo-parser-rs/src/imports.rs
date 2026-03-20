//! Import extraction: `use` statements → IMPORTS edges.

use std::path::Path;

use anyhow::{Context, Result};

use crate::graph::{CodeGraph, Edge};

/// Extract IMPORTS edges from a file's `use` statements.
///
/// Only emits edges for internal imports (use crate::, use super::, use self::)
/// where the target node exists in the graph.
pub fn extract_imports(
    graph: &mut CodeGraph,
    file: &Path,
    module_id: &str,
    crate_name: &str,
) -> Result<()> {
    let content = std::fs::read_to_string(file)
        .with_context(|| format!("Failed to read {}", file.display()))?;
    let syntax = syn::parse_file(&content)
        .with_context(|| format!("Failed to parse {}", file.display()))?;

    let mut targets = Vec::new();
    for item in &syntax.items {
        if let syn::Item::Use(use_item) = item {
            collect_import_targets(&use_item.tree, "", module_id, crate_name, &mut targets);
        }
    }

    // Only emit edges where the target exists in the graph.
    for target_id in targets {
        if graph.has_node(&target_id) && target_id != module_id {
            graph.add_edge(Edge {
                source: module_id.to_string(),
                target: target_id,
                kind: "imports",
            });
        } else {
            // Try the parent module (e.g., `use crate::graph::Graph` → check `crate.graph`).
            if let Some((parent, _)) = target_id.rsplit_once('.') {
                if graph.has_node(parent) && parent != module_id {
                    graph.add_edge(Edge {
                        source: module_id.to_string(),
                        target: parent.to_string(),
                        kind: "imports",
                    });
                }
            }
        }
    }

    Ok(())
}

/// Walk a UseTree and collect all resolved target IDs.
fn collect_import_targets(
    tree: &syn::UseTree,
    prefix: &str,
    module_id: &str,
    crate_name: &str,
    targets: &mut Vec<String>,
) {
    match tree {
        syn::UseTree::Path(p) => {
            let seg = p.ident.to_string();
            let new_prefix = if prefix.is_empty() {
                seg
            } else {
                format!("{prefix}::{seg}")
            };
            collect_import_targets(&p.tree, &new_prefix, module_id, crate_name, targets);
        }
        syn::UseTree::Name(n) => {
            let name = n.ident.to_string();
            let full_path = if prefix.is_empty() {
                name
            } else {
                format!("{prefix}::{name}")
            };
            if let Some(resolved) = resolve_rust_path(&full_path, module_id, crate_name) {
                targets.push(resolved);
            }
        }
        syn::UseTree::Rename(r) => {
            let orig = r.ident.to_string();
            let full_path = if prefix.is_empty() {
                orig
            } else {
                format!("{prefix}::{orig}")
            };
            if let Some(resolved) = resolve_rust_path(&full_path, module_id, crate_name) {
                targets.push(resolved);
            }
        }
        syn::UseTree::Glob(_) => {
            // `use foo::*` → import the module itself.
            if !prefix.is_empty() {
                if let Some(resolved) = resolve_rust_path(prefix, module_id, crate_name) {
                    targets.push(resolved);
                }
            }
        }
        syn::UseTree::Group(g) => {
            for tree in &g.items {
                collect_import_targets(tree, prefix, module_id, crate_name, targets);
            }
        }
    }
}

/// Resolve a Rust use path (with :: separators) to a dotted node ID.
fn resolve_rust_path(path: &str, module_id: &str, crate_name: &str) -> Option<String> {
    let parts: Vec<&str> = path.split("::").collect();
    if parts.is_empty() {
        return None;
    }

    match parts[0] {
        "crate" => {
            let rest = &parts[1..];
            if rest.is_empty() {
                Some(crate_name.to_string())
            } else {
                Some(format!("{}.{}", crate_name, rest.join(".")))
            }
        }
        "super" => {
            let parent = module_id.rsplit_once('.').map(|(p, _)| p).unwrap_or(module_id);
            let rest = &parts[1..];
            if rest.is_empty() {
                Some(parent.to_string())
            } else {
                Some(format!("{}.{}", parent, rest.join(".")))
            }
        }
        "self" => {
            let rest = &parts[1..];
            if rest.is_empty() {
                Some(module_id.to_string())
            } else {
                Some(format!("{}.{}", module_id, rest.join(".")))
            }
        }
        _ => {
            // External crate — not internal.
            None
        }
    }
}
