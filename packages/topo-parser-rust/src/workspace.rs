//! Workspace loading via rust-analyzer's semantic analysis engine.
//!
//! Wraps `ra_ap_load_cargo` to produce a `RootDatabase` + `Vfs` pair
//! that all downstream extraction phases query.

use std::path::Path;

use anyhow::{Context, Result};
use ra_ap_hir as hir;
use ra_ap_ide_db::RootDatabase;
use ra_ap_load_cargo::{LoadCargoConfig, ProcMacroServerChoice, load_workspace_at};
use ra_ap_project_model::{CargoConfig, RustLibSource};
use ra_ap_vfs::Vfs;

/// Load a Rust project into rust-analyzer's semantic database.
///
/// Returns the `RootDatabase` and `Vfs` (maps `FileId` → filesystem paths).
/// All subsequent HIR queries go through the database.
pub fn load_project(root: &Path) -> Result<(RootDatabase, Vfs)> {
    let root = dunce::canonicalize(root)
        .unwrap_or_else(|_| root.to_path_buf());

    // Find the Cargo.toml — could be at root or root/Cargo.toml.
    let manifest = if root.join("Cargo.toml").is_file() {
        root.join("Cargo.toml")
    } else if root.file_name().map_or(false, |n| n == "Cargo.toml") {
        root.clone()
    } else {
        anyhow::bail!("No Cargo.toml found at {}", root.display());
    };

    let cargo_config = CargoConfig {
        sysroot: Some(RustLibSource::Discover),
        ..Default::default()
    };

    let load_config = LoadCargoConfig {
        load_out_dirs_from_check: false,
        with_proc_macro_server: ProcMacroServerChoice::None,
        prefill_caches: false,
        proc_macro_processes: 0,
    };

    let (db, vfs, _proc_macro) = load_workspace_at(
        manifest.as_path(),
        &cargo_config,
        &load_config,
        &|_msg| {}, // progress callback — silent
    )
    .with_context(|| format!("Failed to load workspace at {}", root.display()))?;

    Ok((db, vfs))
}

/// Find all workspace-member crates in the loaded database.
///
/// Filters out sysroot and external dependency crates by checking whether
/// their root file lives under the project root directory.
pub fn workspace_crates(
    db: &RootDatabase,
    vfs: &Vfs,
    project_root: &Path,
) -> Vec<hir::Crate> {
    let root_str = project_root.to_string_lossy();
    hir::Crate::all(db)
        .into_iter()
        .filter(|krate| {
            let root_file = krate.root_file(db);
            let vfs_path = vfs.file_path(root_file);
            match vfs_path.as_path() {
                Some(abs_path) => format!("{abs_path}").starts_with(&*root_str),
                None => false,
            }
        })
        .collect()
}
