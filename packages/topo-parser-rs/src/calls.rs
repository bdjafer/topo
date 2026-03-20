//! Call resolution: extract function/method call sites → CALLS edges.
//!
//! Two tiers of resolution:
//! - Tier 1: Free function calls resolved via import map.
//! - Tier 2: self.method() calls resolved within impl block context.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use anyhow::{Context, Result};
use syn::visit::Visit;

use crate::graph::{CodeGraph, Edge};

/// Extract CALLS edges from a file.
///
/// Walks all function/method bodies, finds call sites, and resolves
/// them against known nodes in the graph.
pub fn extract_calls(
    graph: &mut CodeGraph,
    file: &Path,
    module_id: &str,
    crate_name: &str,
    import_map: &HashMap<String, String>,
) -> Result<()> {
    let content = std::fs::read_to_string(file)
        .with_context(|| format!("Failed to read {}", file.display()))?;
    let syntax = syn::parse_file(&content)
        .with_context(|| format!("Failed to parse {}", file.display()))?;

    // Build a set of all known function node IDs for fast lookup (owned).
    let known_functions: HashSet<String> = graph
        .nodes
        .iter()
        .filter(|n| n.kind == "function")
        .map(|n| n.id.clone())
        .collect();

    let mut edges = Vec::new();

    for item in &syntax.items {
        extract_calls_from_item(
            item,
            module_id,
            None,
            crate_name,
            &known_functions,
            import_map,
            &mut edges,
        );
    }

    // Deduplicate and add edges.
    edges.sort();
    edges.dedup();
    for (source, target) in edges {
        if known_functions.contains(&target) && source != target {
            graph.add_edge(Edge {
                source,
                target,
                kind: "calls",
            });
        }
    }

    Ok(())
}

fn extract_calls_from_item(
    item: &syn::Item,
    module_id: &str,
    impl_context: Option<&str>,
    crate_name: &str,
    known_functions: &HashSet<String>,
    import_map: &HashMap<String, String>,
    edges: &mut Vec<(String, String)>,
) {
    match item {
        syn::Item::Fn(f) => {
            let fn_name = f.sig.ident.to_string();
            let caller_id = if let Some(ctx) = impl_context {
                format!("{ctx}.{fn_name}")
            } else {
                format!("{module_id}.{fn_name}")
            };
            let mut visitor = CallVisitor {
                caller_id,
                module_id: module_id.to_string(),
                impl_context: impl_context.map(|s| s.to_string()),
                crate_name: crate_name.to_string(),
                known_functions,
                import_map,
                edges,
            };
            visitor.visit_block(&f.block);
        }

        syn::Item::Impl(imp) => {
            let type_name = extract_type_name(&imp.self_ty);
            let Some(type_name) = type_name else { return };
            let impl_ctx = format!("{module_id}.{type_name}");

            for impl_item in &imp.items {
                if let syn::ImplItem::Fn(method) = impl_item {
                    let method_name = method.sig.ident.to_string();
                    let caller_id = format!("{impl_ctx}.{method_name}");
                    let mut visitor = CallVisitor {
                        caller_id,
                        module_id: module_id.to_string(),
                        impl_context: Some(impl_ctx.clone()),
                        crate_name: crate_name.to_string(),
                        known_functions,
                        import_map,
                        edges,
                    };
                    visitor.visit_block(&method.block);
                }
            }
        }

        syn::Item::Mod(m) => {
            if let Some((_, items)) = &m.content {
                let mod_name = m.ident.to_string();
                let child_module_id = format!("{module_id}.{mod_name}");
                for item in items {
                    extract_calls_from_item(
                        item,
                        &child_module_id,
                        None,
                        crate_name,
                        known_functions,
                        import_map,
                        edges,
                    );
                }
            }
        }

        _ => {}
    }
}

struct CallVisitor<'a> {
    caller_id: String,
    module_id: String,
    impl_context: Option<String>,
    crate_name: String,
    known_functions: &'a HashSet<String>,
    import_map: &'a HashMap<String, String>,
    edges: &'a mut Vec<(String, String)>,
}

impl<'a, 'ast> Visit<'ast> for CallVisitor<'a> {
    fn visit_expr_call(&mut self, node: &'ast syn::ExprCall) {
        if let syn::Expr::Path(ep) = &*node.func {
            let segments: Vec<String> = ep.path.segments.iter().map(|s| s.ident.to_string()).collect();
            if let Some(resolved) = self.resolve_call_path(&segments) {
                self.edges.push((self.caller_id.clone(), resolved));
            }
        }
        syn::visit::visit_expr_call(self, node);
    }

    fn visit_expr_method_call(&mut self, node: &'ast syn::ExprMethodCall) {
        let method_name = node.method.to_string();

        if is_self_expr(&node.receiver) {
            if let Some(ref impl_ctx) = self.impl_context {
                let target = format!("{impl_ctx}.{method_name}");
                self.edges.push((self.caller_id.clone(), target));
            }
        }
        syn::visit::visit_expr_method_call(self, node);
    }
}

impl<'a> CallVisitor<'a> {
    fn resolve_call_path(&self, segments: &[String]) -> Option<String> {
        if segments.is_empty() {
            return None;
        }

        if segments.len() == 1 {
            let name = &segments[0];

            // Check import map first.
            if let Some(resolved) = self.import_map.get(name.as_str()) {
                return Some(resolved.clone());
            }

            // Check same-module function.
            let local = format!("{}.{}", self.module_id, name);
            if self.known_functions.contains(&local) {
                return Some(local);
            }

            // Check same impl context.
            if let Some(ref ctx) = self.impl_context {
                let associated = format!("{ctx}.{name}");
                if self.known_functions.contains(&associated) {
                    return Some(associated);
                }
            }

            return None;
        }

        // Multi-segment: try imports first.
        let first = &segments[0];
        if let Some(resolved_base) = self.import_map.get(first.as_str()) {
            let rest: Vec<&str> = segments[1..].iter().map(|s| s.as_str()).collect();
            let full = format!("{}.{}", resolved_base, rest.join("."));
            if self.known_functions.contains(&full) {
                return Some(full);
            }
        }

        // Try crate-relative.
        let dotted = segments.join(".");
        let full = format!("{}.{}", self.crate_name, dotted);
        if self.known_functions.contains(&full) {
            return Some(full);
        }

        // Try module-relative.
        let full = format!("{}.{}", self.module_id, dotted);
        if self.known_functions.contains(&full) {
            return Some(full);
        }

        None
    }
}

fn extract_type_name(ty: &syn::Type) -> Option<String> {
    match ty {
        syn::Type::Path(tp) => tp.path.segments.last().map(|seg| seg.ident.to_string()),
        _ => None,
    }
}

fn is_self_expr(expr: &syn::Expr) -> bool {
    matches!(expr, syn::Expr::Path(ep) if ep.path.is_ident("self"))
}
