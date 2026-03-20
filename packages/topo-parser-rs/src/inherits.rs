//! Inherits extraction: `impl Trait for Struct` and `trait Sub: Super`
//! → INHERITS edges.

use std::path::Path;

use anyhow::{Context, Result};

use crate::graph::{CodeGraph, Edge};

/// Extract INHERITS edges from a file.
///
/// - `impl Trait for Struct` → INHERITS edge from Struct to Trait
/// - `trait Sub: Super` → INHERITS edge from Sub to Super
///
/// Only emits edges where both source and target exist in the graph.
pub fn extract_inherits(
    graph: &mut CodeGraph,
    file: &Path,
    module_id: &str,
) -> Result<()> {
    let content = std::fs::read_to_string(file)
        .with_context(|| format!("Failed to read {}", file.display()))?;
    let syntax = syn::parse_file(&content)
        .with_context(|| format!("Failed to parse {}", file.display()))?;

    let mut edges = Vec::new();

    for item in &syntax.items {
        match item {
            syn::Item::Impl(imp) => {
                // Only trait impls: `impl Trait for Type`.
                if let Some((_, trait_path, _)) = &imp.trait_ {
                    let trait_name = extract_path_name(trait_path);
                    let type_name = extract_type_name(&imp.self_ty);

                    if let (Some(trait_name), Some(type_name)) = (trait_name, type_name) {
                        let struct_id = format!("{module_id}.{type_name}");
                        let trait_id = format!("{module_id}.{trait_name}");
                        edges.push((struct_id, trait_id));
                    }
                }
            }

            syn::Item::Trait(t) => {
                let sub_name = t.ident.to_string();
                let sub_id = format!("{module_id}.{sub_name}");

                // Supertrait bounds: `trait Sub: Super + OtherTrait`.
                for bound in &t.supertraits {
                    if let syn::TypeParamBound::Trait(tb) = bound {
                        if let Some(super_name) = extract_path_name(&tb.path) {
                            let super_id = format!("{module_id}.{super_name}");
                            edges.push((sub_id.clone(), super_id));
                        }
                    }
                }
            }

            _ => {}
        }
    }

    // Only emit edges where both nodes exist in the graph.
    for (source, target) in edges {
        if graph.has_node(&source) && graph.has_node(&target) && source != target {
            graph.add_edge(Edge {
                source,
                target,
                kind: "inherits",
            });
        }
    }

    Ok(())
}

fn extract_path_name(path: &syn::Path) -> Option<String> {
    path.segments.last().map(|seg| seg.ident.to_string())
}

fn extract_type_name(ty: &syn::Type) -> Option<String> {
    match ty {
        syn::Type::Path(tp) => tp.path.segments.last().map(|seg| seg.ident.to_string()),
        _ => None,
    }
}
