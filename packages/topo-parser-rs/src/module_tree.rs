//! Rust module tree resolution: map source files to fully qualified module IDs.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use crate::discovery::CrateInfo;

/// Map each `.rs` file in a crate to its fully qualified module ID.
///
/// Resolution follows Rust's module system:
/// - `src/lib.rs` or `src/main.rs` → crate root module
/// - `src/<name>.rs` → `<crate>.<name>`
/// - `src/<name>/mod.rs` → `<crate>.<name>`
/// - Nested modules follow the same pattern recursively.
pub fn resolve_module_tree(krate: &CrateInfo) -> Result<HashMap<PathBuf, String>> {
    let mut file_to_module: HashMap<PathBuf, String> = HashMap::new();

    // Find the crate entry point.
    let lib_rs = krate.src_dir.join("lib.rs");
    let main_rs = krate.src_dir.join("main.rs");

    let entry = if lib_rs.is_file() {
        lib_rs
    } else if main_rs.is_file() {
        main_rs
    } else {
        anyhow::bail!(
            "No lib.rs or main.rs in {}",
            krate.src_dir.display()
        );
    };

    // Map entry point to crate root module.
    file_to_module.insert(entry.clone(), krate.name.clone());

    // Resolve modules recursively starting from the entry point.
    resolve_mods_in_file(&entry, &krate.name, &krate.src_dir, &mut file_to_module)?;

    Ok(file_to_module)
}

/// Parse a file for `mod <name>;` declarations and resolve them to files.
fn resolve_mods_in_file(
    file: &Path,
    module_id: &str,
    src_dir: &Path,
    file_to_module: &mut HashMap<PathBuf, String>,
) -> Result<()> {
    let content = std::fs::read_to_string(file)
        .with_context(|| format!("Failed to read {}", file.display()))?;

    let syntax = syn::parse_file(&content)
        .with_context(|| format!("Failed to parse {}", file.display()))?;

    let parent_dir = file.parent().unwrap();

    for item in &syntax.items {
        if let syn::Item::Mod(item_mod) = item {
            let mod_name = item_mod.ident.to_string();
            let child_module_id = format!("{module_id}.{mod_name}");

            if item_mod.content.is_some() {
                // Inline module: `mod foo { ... }` — no file to resolve.
                file_to_module
                    .entry(file.to_path_buf())
                    .or_insert_with(|| module_id.to_string());
                // We'll extract inline module items during the items phase.
                continue;
            }

            // File-based module: `mod foo;`
            // Try <parent_dir>/<name>.rs, then <parent_dir>/<name>/mod.rs.
            let candidate_file = parent_dir.join(format!("{mod_name}.rs"));
            let candidate_dir = parent_dir.join(&mod_name).join("mod.rs");

            let resolved = if candidate_file.is_file() {
                candidate_file
            } else if candidate_dir.is_file() {
                candidate_dir
            } else {
                // cfg-gated or otherwise missing — skip silently.
                continue;
            };

            // Resolve path relative to src_dir for consistency.
            let resolved = if let Ok(canonical) = resolved.canonicalize() {
                canonical
            } else {
                resolved
            };

            file_to_module.insert(resolved.clone(), child_module_id.clone());

            // Recurse into the resolved file.
            resolve_mods_in_file(&resolved, &child_module_id, src_dir, file_to_module)?;
        }
    }

    Ok(())
}
