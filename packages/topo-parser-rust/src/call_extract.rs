//! Call edge extraction using rust-analyzer's Semantics.
//!
//! Walks function bodies and resolves every call expression to its target
//! using full type inference and name resolution. This replaces the old
//! syn-based call resolution that missed 69% of calls.

use std::collections::HashMap;

use ra_ap_hir::{self as hir, HasSource, Semantics};
use ra_ap_ide_db::RootDatabase;
use ra_ap_syntax::{AstNode, ast};
use ra_ap_hir::EditionedFileId;

use crate::graph::{CodeGraph, Edge};
use crate::hir_walk::HirEntity;

/// Extract CALLS edges for every function in a crate.
///
/// Groups functions by file to avoid redundant `sema.parse()` calls.
pub fn extract_calls(
    db: &RootDatabase,
    sema: &Semantics<'_, RootDatabase>,
    krate: &hir::Crate,
    graph: &mut CodeGraph,
    entity_map: &HashMap<HirEntity, String>,
) {
    // Collect functions belonging to this crate, grouped by file.
    let mut by_file: HashMap<EditionedFileId, Vec<(hir::Function, String, ast::BlockExpr)>> =
        HashMap::new();

    for (entity, id) in entity_map {
        let HirEntity::Function(func) = entity else {
            continue;
        };
        if func.module(db).krate(db) != *krate {
            continue;
        }
        let Some(source) = func.source(db) else {
            continue;
        };
        let Some(file_id) = source.file_id.file_id() else {
            continue;
        };
        let Some(body) = source.value.body() else {
            continue;
        };
        by_file
            .entry(file_id)
            .or_default()
            .push((*func, id.clone(), body));
    }

    // Process each file once.
    for (file_id, funcs) in &by_file {
        let sema_file = sema.parse(*file_id);
        let sema_root = sema_file.syntax();

        // Index all BlockExpr nodes by text range for O(1) lookup.
        let blocks: HashMap<_, _> = sema_root
            .descendants()
            .filter_map(ast::BlockExpr::cast)
            .map(|b| (b.syntax().text_range(), b))
            .collect();

        for (_func, caller_id, body) in funcs {
            let fn_range = body.syntax().text_range();
            let Some(sema_body) = blocks.get(&fn_range) else {
                continue;
            };

            for node in sema_body.syntax().descendants() {
                if let Some(call_expr) = ast::CallExpr::cast(node.clone()) {
                    if let Some(expr) = call_expr.expr() {
                        resolve_call_expr(sema, &expr, caller_id, entity_map, graph);
                    }
                }
                if let Some(method_call) = ast::MethodCallExpr::cast(node.clone()) {
                    resolve_method_call(sema, &method_call, caller_id, entity_map, graph);
                }
            }
        }
    }
}

/// Resolve a function call expression to its target.
fn resolve_call_expr(
    sema: &Semantics<'_, RootDatabase>,
    expr: &ast::Expr,
    caller_id: &str,
    entity_map: &HashMap<HirEntity, String>,
    graph: &mut CodeGraph,
) {
    // Try path expression resolution (most common case).
    if let Some(path_expr) = ast::PathExpr::cast(expr.syntax().clone()) {
        if let Some(resolved) = sema.resolve_path(&path_expr.path().unwrap()) {
            if let hir::PathResolution::Def(hir::ModuleDef::Function(f)) = resolved {
                if let Some(target_id) = entity_map.get(&HirEntity::Function(f)) {
                    if target_id != caller_id {
                        graph.add_edge(Edge {
                            source: caller_id.to_string(),
                            target: target_id.clone(),
                            kind: "calls",
                        });
                    }
                }
            }
        }
    }
}

/// Resolve a method call to its target function.
fn resolve_method_call(
    sema: &Semantics<'_, RootDatabase>,
    method_call: &ast::MethodCallExpr,
    caller_id: &str,
    entity_map: &HashMap<HirEntity, String>,
    graph: &mut CodeGraph,
) {
    if let Some(func) = sema.resolve_method_call(method_call) {
        if let Some(target_id) = entity_map.get(&HirEntity::Function(func)) {
            if target_id != caller_id {
                graph.add_edge(Edge {
                    source: caller_id.to_string(),
                    target: target_id.clone(),
                    kind: "calls",
                });
            }
        }
    }
}
