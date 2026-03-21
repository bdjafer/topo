//! HIR tree walking: extract nodes, CONTAINS, IMPORTS, and INHERITS edges.
//!
//! Replaces the old syn-based items.rs, imports.rs, and inherits.rs with
//! rust-analyzer's fully resolved semantic model.

use std::collections::HashMap;

use ra_ap_hir::{self as hir, HasSource};
use ra_ap_ide_db::RootDatabase;
use ra_ap_syntax::AstNode;
use ra_ap_vfs::Vfs;

use crate::graph::{CodeGraph, Edge, Node};
use crate::source_map::SourceMapper;

/// Key for mapping HIR entities to their node IDs in the graph.
#[derive(Clone, Copy, Hash, Eq, PartialEq)]
pub enum HirEntity {
    Module(hir::Module),
    Function(hir::Function),
    Struct(hir::Struct),
    Enum(hir::Enum),
    Trait(hir::Trait),
}

/// Extract all nodes and CONTAINS edges from a crate.
pub fn extract_nodes(
    db: &RootDatabase,
    vfs: &Vfs,
    mapper: &SourceMapper,
    krate: &hir::Crate,
    graph: &mut CodeGraph,
    entity_map: &mut HashMap<HirEntity, String>,
) {
    let root_module = krate.root_module(db);
    let crate_name = krate
        .display_name(db)
        .map(|n| n.to_string().replace('-', "_"))
        .unwrap_or_else(|| "unknown".to_string());

    extract_module(db, vfs, mapper, &root_module, &crate_name, None, graph, entity_map);
}

/// Recursively extract a module and its contents.
fn extract_module(
    db: &RootDatabase,
    vfs: &Vfs,
    mapper: &SourceMapper,
    module: &hir::Module,
    module_id: &str,
    parent_id: Option<&str>,
    graph: &mut CodeGraph,
    entity_map: &mut HashMap<HirEntity, String>,
) {
    // Get file + line for this module.
    let (file_str, line) = module_source_location(db, vfs, mapper, module);

    // Add module node.
    let name = module
        .name(db)
        .map(|n| n.as_str().to_string())
        .unwrap_or_else(|| module_id.rsplit('.').next().unwrap_or(module_id).to_string());

    graph.add_node(Node {
        id: module_id.to_string(),
        kind: "module",
        file: file_str,
        line,
        name: Some(name),
    });
    entity_map.insert(HirEntity::Module(*module), module_id.to_string());

    // CONTAINS edge from parent.
    if let Some(parent) = parent_id {
        graph.add_edge(Edge {
            source: parent.to_string(),
            target: module_id.to_string(),
            kind: "contains",
        });
    }

    // Process declarations in this module.
    for def in module.declarations(db) {
        match def {
            hir::ModuleDef::Module(child) => {
                let child_name = child
                    .name(db)
                    .map(|n| n.as_str().to_string())
                    .unwrap_or_else(|| "unnamed".to_string());
                let child_id = format!("{module_id}.{child_name}");
                extract_module(db, vfs, mapper, &child, &child_id, Some(module_id), graph, entity_map);
            }
            hir::ModuleDef::Function(f) => {
                extract_function(db, vfs, mapper, &f, module_id, module_id, graph, entity_map);
            }
            hir::ModuleDef::Adt(adt) => match adt {
                hir::Adt::Struct(s) => {
                    extract_struct(db, vfs, mapper, &s, module_id, graph, entity_map);
                }
                hir::Adt::Enum(e) => {
                    extract_enum(db, vfs, mapper, &e, module_id, graph, entity_map);
                }
                hir::Adt::Union(_) => {}
            },
            hir::ModuleDef::Trait(t) => {
                extract_trait(db, vfs, mapper, &t, module_id, graph, entity_map);
            }
            _ => {}
        }
    }

    // Process impl blocks in this module.
    for imp in module.impl_defs(db) {
        extract_impl_methods(db, vfs, mapper, &imp, module_id, graph, entity_map);
    }
}

/// Extract a free function node.
fn extract_function(
    db: &RootDatabase,
    vfs: &Vfs,
    mapper: &SourceMapper,
    func: &hir::Function,
    parent_id: &str,
    module_id: &str,
    graph: &mut CodeGraph,
    entity_map: &mut HashMap<HirEntity, String>,
) {
    let name = func.name(db).as_str().to_string();
    let fn_id = format!("{parent_id}.{name}");

    let (file_str, line) = function_source_location(db, vfs, mapper, func);

    graph.add_node(Node {
        id: fn_id.clone(),
        kind: "function",
        file: file_str,
        line,
        name: Some(name),
    });
    entity_map.insert(HirEntity::Function(*func), fn_id.clone());

    // CONTAINS: module → function.
    graph.add_edge(Edge {
        source: module_id.to_string(),
        target: fn_id,
        kind: "contains",
    });
}

/// Extract a struct as a "class" node.
fn extract_struct(
    db: &RootDatabase,
    vfs: &Vfs,
    mapper: &SourceMapper,
    s: &hir::Struct,
    module_id: &str,
    graph: &mut CodeGraph,
    entity_map: &mut HashMap<HirEntity, String>,
) {
    let name = s.name(db).as_str().to_string();
    let id = format!("{module_id}.{name}");

    let (file_str, line) = struct_source_location(db, vfs, mapper, s);

    graph.add_node(Node {
        id: id.clone(),
        kind: "class",
        file: file_str,
        line,
        name: Some(name),
    });
    entity_map.insert(HirEntity::Struct(*s), id.clone());

    graph.add_edge(Edge {
        source: module_id.to_string(),
        target: id,
        kind: "contains",
    });
}

/// Extract an enum as a "class" node.
fn extract_enum(
    db: &RootDatabase,
    vfs: &Vfs,
    mapper: &SourceMapper,
    e: &hir::Enum,
    module_id: &str,
    graph: &mut CodeGraph,
    entity_map: &mut HashMap<HirEntity, String>,
) {
    let name = e.name(db).as_str().to_string();
    let id = format!("{module_id}.{name}");

    let (file_str, line) = enum_source_location(db, vfs, mapper, e);

    graph.add_node(Node {
        id: id.clone(),
        kind: "class",
        file: file_str,
        line,
        name: Some(name),
    });
    entity_map.insert(HirEntity::Enum(*e), id.clone());

    graph.add_edge(Edge {
        source: module_id.to_string(),
        target: id,
        kind: "contains",
    });
}

/// Extract a trait as a "class" node, including its declared methods.
fn extract_trait(
    db: &RootDatabase,
    vfs: &Vfs,
    mapper: &SourceMapper,
    t: &hir::Trait,
    module_id: &str,
    graph: &mut CodeGraph,
    entity_map: &mut HashMap<HirEntity, String>,
) {
    let name = t.name(db).as_str().to_string();
    let trait_id = format!("{module_id}.{name}");

    let (file_str, line) = trait_source_location(db, vfs, mapper, t);

    graph.add_node(Node {
        id: trait_id.clone(),
        kind: "class",
        file: file_str,
        line,
        name: Some(name),
    });
    entity_map.insert(HirEntity::Trait(*t), trait_id.clone());

    graph.add_edge(Edge {
        source: module_id.to_string(),
        target: trait_id.clone(),
        kind: "contains",
    });

    // Trait methods as function nodes.
    for item in t.items(db) {
        if let hir::AssocItem::Function(f) = item {
            extract_function(db, vfs, mapper, &f, &trait_id, module_id, graph, entity_map);
        }
    }
}

/// Extract methods from an impl block, parenting them to the target type.
fn extract_impl_methods(
    db: &RootDatabase,
    vfs: &Vfs,
    mapper: &SourceMapper,
    imp: &hir::Impl,
    module_id: &str,
    graph: &mut CodeGraph,
    entity_map: &mut HashMap<HirEntity, String>,
) {
    // Skip standard-library trait impls — their methods represent
    // language-level contracts, not architectural coupling. The import
    // graph already captures type dependencies.
    if let Some(trait_) = imp.trait_(db) {
        let trait_crate_name = trait_
            .module(db)
            .krate(db)
            .display_name(db)
            .map(|n| n.to_string())
            .unwrap_or_default();
        if matches!(trait_crate_name.as_str(), "core" | "std" | "alloc") {
            return;
        }
    }

    let self_ty = imp.self_ty(db);
    let type_name = self_ty.as_adt().map(|adt| match adt {
        hir::Adt::Struct(s) => s.name(db).as_str().to_string(),
        hir::Adt::Enum(e) => e.name(db).as_str().to_string(),
        hir::Adt::Union(u) => u.name(db).as_str().to_string(),
    });

    let parent_id = match &type_name {
        Some(name) => format!("{module_id}.{name}"),
        None => module_id.to_string(),
    };

    for item in imp.items(db) {
        if let hir::AssocItem::Function(f) = item {
            let name = f.name(db).as_str().to_string();
            let fn_id = format!("{parent_id}.{name}");

            // Skip if already added (e.g., trait method with default body).
            if entity_map.contains_key(&HirEntity::Function(f)) {
                continue;
            }

            let (file_str, line) = function_source_location(db, vfs, mapper, &f);

            graph.add_node(Node {
                id: fn_id.clone(),
                kind: "function",
                file: file_str,
                line,
                name: Some(name),
            });
            entity_map.insert(HirEntity::Function(f), fn_id.clone());

            graph.add_edge(Edge {
                source: parent_id.clone(),
                target: fn_id,
                kind: "contains",
            });
        }
    }
}

// ── Source location helpers ──────────────────────────────────────────────

fn module_source_location(
    db: &RootDatabase,
    vfs: &Vfs,
    mapper: &SourceMapper,
    module: &hir::Module,
) -> (Option<String>, Option<u32>) {
    let source = module.definition_source(db);
    let file_id = source.file_id.file_id();
    let file_str = file_id.and_then(|fid| mapper.file_path(vfs, db, fid));
    let line = match &source.value {
        hir::ModuleSource::Module(m) => {
            file_id.and_then(|fid| {
                let offset = m.syntax().text_range().start();
                        mapper.line_number(db, fid, offset)
            })
        }
        _ => Some(1),
    };
    (file_str, line)
}

fn function_source_location(
    db: &RootDatabase,
    vfs: &Vfs,
    mapper: &SourceMapper,
    func: &hir::Function,
) -> (Option<String>, Option<u32>) {
    let Some(src) = func.source(db) else { return (None, None) };
    let file_id = src.file_id.file_id();
    let file_str = file_id.and_then(|fid| mapper.file_path(vfs, db, fid));
    let line = file_id.and_then(|fid| {
        let offset = src.value.syntax().text_range().start();
                mapper.line_number(db, fid, offset)
    });
    (file_str, line)
}

fn struct_source_location(
    db: &RootDatabase,
    vfs: &Vfs,
    mapper: &SourceMapper,
    s: &hir::Struct,
) -> (Option<String>, Option<u32>) {
    let Some(src) = s.source(db) else { return (None, None) };
    let file_id = src.file_id.file_id();
    let file_str = file_id.and_then(|fid| mapper.file_path(vfs, db, fid));
    let line = file_id.and_then(|fid| {
        let offset = src.value.syntax().text_range().start();
                mapper.line_number(db, fid, offset)
    });
    (file_str, line)
}

fn enum_source_location(
    db: &RootDatabase,
    vfs: &Vfs,
    mapper: &SourceMapper,
    e: &hir::Enum,
) -> (Option<String>, Option<u32>) {
    let Some(src) = e.source(db) else { return (None, None) };
    let file_id = src.file_id.file_id();
    let file_str = file_id.and_then(|fid| mapper.file_path(vfs, db, fid));
    let line = file_id.and_then(|fid| {
        let offset = src.value.syntax().text_range().start();
                mapper.line_number(db, fid, offset)
    });
    (file_str, line)
}

fn trait_source_location(
    db: &RootDatabase,
    vfs: &Vfs,
    mapper: &SourceMapper,
    t: &hir::Trait,
) -> (Option<String>, Option<u32>) {
    let Some(src) = t.source(db) else { return (None, None) };
    let file_id = src.file_id.file_id();
    let file_str = file_id.and_then(|fid| mapper.file_path(vfs, db, fid));
    let line = file_id.and_then(|fid| {
        let offset = src.value.syntax().text_range().start();
                mapper.line_number(db, fid, offset)
    });
    (file_str, line)
}

// ── Import extraction ───────────────────────────────────────────────────

/// Extract IMPORTS edges for all modules in a crate.
pub fn extract_imports(
    db: &RootDatabase,
    krate: &hir::Crate,
    graph: &mut CodeGraph,
    entity_map: &HashMap<HirEntity, String>,
) {
    let root = krate.root_module(db);
    extract_module_imports(db, &root, graph, entity_map);
}

fn extract_module_imports(
    db: &RootDatabase,
    module: &hir::Module,
    graph: &mut CodeGraph,
    entity_map: &HashMap<HirEntity, String>,
) {
    let module_id = match entity_map.get(&HirEntity::Module(*module)) {
        Some(id) => id.clone(),
        None => return,
    };

    // Walk scope to find imported items from other modules.
    for (_name, scope_def) in module.scope(db, None) {
        let target_module = match scope_def {
            hir::ScopeDef::ModuleDef(hir::ModuleDef::Function(f)) => Some(f.module(db)),
            hir::ScopeDef::ModuleDef(hir::ModuleDef::Adt(adt)) => Some(match adt {
                hir::Adt::Struct(s) => s.module(db),
                hir::Adt::Enum(e) => e.module(db),
                hir::Adt::Union(u) => u.module(db),
            }),
            hir::ScopeDef::ModuleDef(hir::ModuleDef::Trait(t)) => Some(t.module(db)),
            hir::ScopeDef::ModuleDef(hir::ModuleDef::Module(m)) => {
                if *module != m {
                    Some(m)
                } else {
                    None
                }
            }
            _ => None,
        };

        if let Some(def_module) = target_module {
            if def_module != *module {
                if let Some(target_id) = entity_map.get(&HirEntity::Module(def_module)) {
                    if *target_id != module_id {
                        graph.add_edge(Edge {
                            source: module_id.clone(),
                            target: target_id.clone(),
                            kind: "imports",
                        });
                    }
                }
            }
        }
    }

    // Recurse into child modules.
    for def in module.declarations(db) {
        if let hir::ModuleDef::Module(child) = def {
            extract_module_imports(db, &child, graph, entity_map);
        }
    }
}

// ── Inherits extraction ─────────────────────────────────────────────────

/// Extract INHERITS edges for a crate.
pub fn extract_inherits(
    db: &RootDatabase,
    krate: &hir::Crate,
    graph: &mut CodeGraph,
    entity_map: &HashMap<HirEntity, String>,
) {
    // Trait impls.
    for imp in hir::Impl::all_in_crate(db, *krate) {
        let Some(trait_) = imp.trait_(db) else { continue };
        let trait_id = match entity_map.get(&HirEntity::Trait(trait_)) {
            Some(id) => id.clone(),
            None => continue,
        };

        let self_ty = imp.self_ty(db);
        let self_id = self_ty
            .as_adt()
            .and_then(|adt| match adt {
                hir::Adt::Struct(s) => entity_map.get(&HirEntity::Struct(s)),
                hir::Adt::Enum(e) => entity_map.get(&HirEntity::Enum(e)),
                hir::Adt::Union(_) => None,
            });

        if let Some(self_id) = self_id {
            graph.add_edge(Edge {
                source: self_id.clone(),
                target: trait_id,
                kind: "inherits",
            });
        }
    }

    // Supertraits.
    let root = krate.root_module(db);
    extract_supertraits(db, &root, graph, entity_map);
}

fn extract_supertraits(
    db: &RootDatabase,
    module: &hir::Module,
    graph: &mut CodeGraph,
    entity_map: &HashMap<HirEntity, String>,
) {
    for def in module.declarations(db) {
        match def {
            hir::ModuleDef::Trait(t) => {
                let Some(trait_id) = entity_map.get(&HirEntity::Trait(t)) else { continue };
                for super_trait in t.direct_supertraits(db) {
                    if let Some(super_id) = entity_map.get(&HirEntity::Trait(super_trait)) {
                        graph.add_edge(Edge {
                            source: trait_id.clone(),
                            target: super_id.clone(),
                            kind: "inherits",
                        });
                    }
                }
            }
            hir::ModuleDef::Module(child) => {
                extract_supertraits(db, &child, graph, entity_map);
            }
            _ => {}
        }
    }
}
