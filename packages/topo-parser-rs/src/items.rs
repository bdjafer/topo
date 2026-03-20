//! Item extraction: parse .rs files into nodes (module, class, function)
//! and CONTAINS edges.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use anyhow::{Context, Result};

use crate::graph::{CodeGraph, Edge, Node};

/// Extract all items from a file into the graph.
///
/// - Creates function, struct/enum/trait (class) nodes.
/// - Creates CONTAINS edges from module → items and type → methods.
/// - Resolves impl block methods to their target type.
pub fn extract_items(
    graph: &mut CodeGraph,
    file: &Path,
    module_id: &str,
    relative_file: &str,
) -> Result<()> {
    let content = std::fs::read_to_string(file)
        .with_context(|| format!("Failed to read {}", file.display()))?;
    let syntax = syn::parse_file(&content)
        .with_context(|| format!("Failed to parse {}", file.display()))?;

    // First pass: collect type names defined in this module (for impl resolution).
    let mut local_types: HashSet<String> = HashSet::new();
    for item in &syntax.items {
        match item {
            syn::Item::Struct(s) => { local_types.insert(s.ident.to_string()); }
            syn::Item::Enum(e) => { local_types.insert(e.ident.to_string()); }
            syn::Item::Trait(t) => { local_types.insert(t.ident.to_string()); }
            _ => {}
        }
    }

    // Second pass: extract items.
    for item in &syntax.items {
        extract_item(graph, item, module_id, relative_file, &local_types);
    }

    Ok(())
}

fn extract_item(
    graph: &mut CodeGraph,
    item: &syn::Item,
    parent_id: &str,
    relative_file: &str,
    local_types: &HashSet<String>,
) {
    match item {
        syn::Item::Fn(f) => {
            let name = f.sig.ident.to_string();
            let id = format!("{parent_id}.{name}");
            let line = line_of(&f.sig.fn_token.span);
            graph.add_node(Node {
                id: id.clone(),
                kind: "function",
                file: Some(relative_file.to_string()),
                line: Some(line),
                name: Some(name),
            });
            graph.add_edge(Edge {
                source: parent_id.to_string(),
                target: id,
                kind: "contains",
            });
        }

        syn::Item::Struct(s) => {
            let name = s.ident.to_string();
            let id = format!("{parent_id}.{name}");
            let line = line_of(&s.struct_token.span);
            graph.add_node(Node {
                id: id.clone(),
                kind: "class",
                file: Some(relative_file.to_string()),
                line: Some(line),
                name: Some(name),
            });
            graph.add_edge(Edge {
                source: parent_id.to_string(),
                target: id,
                kind: "contains",
            });
        }

        syn::Item::Enum(e) => {
            let name = e.ident.to_string();
            let id = format!("{parent_id}.{name}");
            let line = line_of(&e.enum_token.span);
            graph.add_node(Node {
                id: id.clone(),
                kind: "class",
                file: Some(relative_file.to_string()),
                line: Some(line),
                name: Some(name),
            });
            graph.add_edge(Edge {
                source: parent_id.to_string(),
                target: id,
                kind: "contains",
            });
        }

        syn::Item::Trait(t) => {
            let name = t.ident.to_string();
            let id = format!("{parent_id}.{name}");
            let line = line_of(&t.trait_token.span);
            graph.add_node(Node {
                id: id.clone(),
                kind: "class",
                file: Some(relative_file.to_string()),
                line: Some(line),
                name: Some(name),
            });
            graph.add_edge(Edge {
                source: parent_id.to_string(),
                target: id.clone(),
                kind: "contains",
            });

            // Extract trait methods as function nodes.
            for trait_item in &t.items {
                if let syn::TraitItem::Fn(method) = trait_item {
                    let method_name = method.sig.ident.to_string();
                    let method_id = format!("{id}.{method_name}");
                    let method_line = line_of(&method.sig.fn_token.span);
                    graph.add_node(Node {
                        id: method_id.clone(),
                        kind: "function",
                        file: Some(relative_file.to_string()),
                        line: Some(method_line),
                        name: Some(method_name),
                    });
                    graph.add_edge(Edge {
                        source: id.clone(),
                        target: method_id,
                        kind: "contains",
                    });
                }
            }
        }

        syn::Item::Impl(imp) => {
            let type_name = extract_type_name(&imp.self_ty);
            let Some(type_name) = type_name else { return };

            // Determine the parent ID for methods.
            // If it's a trait impl (impl Trait for Type), parent to Type.
            // For inherent impl (impl Type), parent to Type.
            let impl_parent = if local_types.contains(&type_name) {
                format!("{parent_id}.{type_name}")
            } else {
                // Foreign type — parent methods under the module with type prefix.
                format!("{parent_id}.{type_name}")
            };

            // Ensure the type node exists (it might be defined in another file).
            if !graph.has_node(&impl_parent) {
                // Create a synthetic class node for the impl target.
                graph.add_node(Node {
                    id: impl_parent.clone(),
                    kind: "class",
                    file: Some(relative_file.to_string()),
                    line: Some(line_of(&imp.impl_token.span)),
                    name: Some(type_name.clone()),
                });
                graph.add_edge(Edge {
                    source: parent_id.to_string(),
                    target: impl_parent.clone(),
                    kind: "contains",
                });
            }

            for impl_item in &imp.items {
                if let syn::ImplItem::Fn(method) = impl_item {
                    let method_name = method.sig.ident.to_string();
                    let method_id = format!("{impl_parent}.{method_name}");
                    let method_line = line_of(&method.sig.fn_token.span);
                    graph.add_node(Node {
                        id: method_id.clone(),
                        kind: "function",
                        file: Some(relative_file.to_string()),
                        line: Some(method_line),
                        name: Some(method_name),
                    });
                    graph.add_edge(Edge {
                        source: impl_parent.clone(),
                        target: method_id,
                        kind: "contains",
                    });
                }
            }
        }

        syn::Item::Mod(m) => {
            // Inline module: `mod foo { ... }`
            if let Some((_, items)) = &m.content {
                let mod_name = m.ident.to_string();
                let mod_id = format!("{parent_id}.{mod_name}");
                let mod_line = line_of(&m.mod_token.span);

                graph.add_node(Node {
                    id: mod_id.clone(),
                    kind: "module",
                    file: Some(relative_file.to_string()),
                    line: Some(mod_line),
                    name: Some(mod_name),
                });
                graph.add_edge(Edge {
                    source: parent_id.to_string(),
                    target: mod_id.clone(),
                    kind: "contains",
                });

                // Collect local types for the inline module.
                let mut inline_types = HashSet::new();
                for item in items {
                    match item {
                        syn::Item::Struct(s) => { inline_types.insert(s.ident.to_string()); }
                        syn::Item::Enum(e) => { inline_types.insert(e.ident.to_string()); }
                        syn::Item::Trait(t) => { inline_types.insert(t.ident.to_string()); }
                        _ => {}
                    }
                }

                for item in items {
                    extract_item(graph, item, &mod_id, relative_file, &inline_types);
                }
            }
            // File-based `mod foo;` is handled by module_tree.rs — no action here.
        }

        _ => {} // Skip const, static, type alias, extern, macro, etc.
    }
}

/// Extract the simple type name from a syn::Type (e.g., `Graph` from `Graph<'a>`).
fn extract_type_name(ty: &syn::Type) -> Option<String> {
    match ty {
        syn::Type::Path(tp) => {
            // Take the last segment's ident (ignoring generics).
            tp.path.segments.last().map(|seg| seg.ident.to_string())
        }
        _ => None,
    }
}

/// Get line number from a proc_macro2::Span.
fn line_of(span: &proc_macro2::Span) -> u32 {
    span.start().line as u32
}

/// Collect all import mappings from a file for use in call resolution.
///
/// Returns a map: short_name → fully_qualified_id.
/// Only tracks `use crate::...` imports (internal imports).
pub fn collect_import_map(
    file: &Path,
    module_id: &str,
    crate_name: &str,
) -> Result<HashMap<String, String>> {
    let content = std::fs::read_to_string(file)
        .with_context(|| format!("Failed to read {}", file.display()))?;
    let syntax = syn::parse_file(&content)
        .with_context(|| format!("Failed to parse {}", file.display()))?;

    let mut imports = HashMap::new();
    for item in &syntax.items {
        if let syn::Item::Use(use_item) = item {
            collect_use_tree(&use_item.tree, "", module_id, crate_name, &mut imports);
        }
    }
    Ok(imports)
}

fn collect_use_tree(
    tree: &syn::UseTree,
    prefix: &str,
    module_id: &str,
    crate_name: &str,
    imports: &mut HashMap<String, String>,
) {
    match tree {
        syn::UseTree::Path(p) => {
            let seg = p.ident.to_string();
            let new_prefix = if prefix.is_empty() {
                seg
            } else {
                format!("{prefix}.{seg}")
            };
            collect_use_tree(&p.tree, &new_prefix, module_id, crate_name, imports);
        }
        syn::UseTree::Name(n) => {
            let name = n.ident.to_string();
            let full_path = if prefix.is_empty() {
                name.clone()
            } else {
                format!("{prefix}.{name}")
            };
            let resolved = resolve_use_path(&full_path, module_id, crate_name);
            if let Some(resolved) = resolved {
                imports.insert(name, resolved);
            }
        }
        syn::UseTree::Rename(r) => {
            let name = r.rename.to_string();
            let orig = r.ident.to_string();
            let full_path = if prefix.is_empty() {
                orig
            } else {
                format!("{prefix}.{orig}")
            };
            let resolved = resolve_use_path(&full_path, module_id, crate_name);
            if let Some(resolved) = resolved {
                imports.insert(name, resolved);
            }
        }
        syn::UseTree::Glob(_) => {
            // `use foo::*` — we can't resolve individual names. Skip.
        }
        syn::UseTree::Group(g) => {
            for tree in &g.items {
                collect_use_tree(tree, prefix, module_id, crate_name, imports);
            }
        }
    }
}

/// Resolve a dotted use path (with `crate`/`super`/`self` prefixes) to a node ID.
/// Returns None for external crate imports.
fn resolve_use_path(path: &str, module_id: &str, crate_name: &str) -> Option<String> {
    let parts: Vec<&str> = path.split('.').collect();
    if parts.is_empty() {
        return None;
    }

    match parts[0] {
        "crate" => {
            // `crate::foo::Bar` → `<crate_name>.foo.Bar`
            let rest = &parts[1..];
            if rest.is_empty() {
                Some(crate_name.to_string())
            } else {
                Some(format!("{}.{}", crate_name, rest.join(".")))
            }
        }
        "super" => {
            // `super::foo` → parent module + foo
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
            // External crate import — not internal. Skip.
            None
        }
    }
}
