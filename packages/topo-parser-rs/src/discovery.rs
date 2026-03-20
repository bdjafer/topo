//! Crate and file discovery for Rust projects.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use walkdir::WalkDir;

/// Information about a single Rust crate.
#[derive(Debug, Clone)]
pub struct CrateInfo {
    /// Crate name with hyphens replaced by underscores (Rust convention).
    pub name: String,
    /// Path to the crate root directory (where Cargo.toml lives).
    pub root: PathBuf,
    /// Path to the `src/` directory.
    pub src_dir: PathBuf,
    /// All `.rs` source files in the crate.
    pub src_files: Vec<PathBuf>,
}

/// Directories to always skip when walking.
const SKIP_DIRS: &[&str] = &["target", ".git", "node_modules", ".venv", "pkg"];

/// Discover all Rust crates under `root`.
///
/// If `root` contains a workspace Cargo.toml, discovers all member crates.
/// If `root` is a single crate, returns just that crate.
/// If `root` contains nested crates (e.g. packages/foo, packages/bar),
/// discovers all of them.
pub fn discover_crates(root: &Path) -> Result<Vec<CrateInfo>> {
    let root = root.canonicalize().context("Cannot canonicalize root")?;
    let mut crates = Vec::new();

    // Check for workspace Cargo.toml.
    let cargo_toml = root.join("Cargo.toml");
    if cargo_toml.is_file() {
        let content = std::fs::read_to_string(&cargo_toml)
            .context("Failed to read Cargo.toml")?;
        let parsed: toml::Value = content.parse()
            .context("Failed to parse Cargo.toml")?;

        if let Some(members) = parsed
            .get("workspace")
            .and_then(|w| w.get("members"))
            .and_then(|m| m.as_array())
        {
            // Workspace: discover each member.
            for member in members {
                if let Some(pattern) = member.as_str() {
                    let member_paths = resolve_glob_pattern(&root, pattern);
                    for member_path in member_paths {
                        if let Ok(info) = crate_info_from_dir(&member_path) {
                            crates.push(info);
                        }
                    }
                }
            }
            if !crates.is_empty() {
                return Ok(crates);
            }
        }

        // Single crate at root.
        if let Ok(info) = crate_info_from_dir(&root) {
            return Ok(vec![info]);
        }
    }

    // Walk to find nested Cargo.toml files.
    let mut seen = HashSet::new();
    for entry in WalkDir::new(&root)
        .follow_links(false)
        .into_iter()
        .filter_entry(|e| {
            let name = e.file_name().to_string_lossy();
            !SKIP_DIRS.contains(&name.as_ref())
        })
    {
        let entry = entry?;
        if entry.file_name() == "Cargo.toml" && entry.path() != cargo_toml {
            let dir = entry.path().parent().unwrap().to_path_buf();
            if seen.insert(dir.clone()) {
                if let Ok(info) = crate_info_from_dir(&dir) {
                    crates.push(info);
                }
            }
        }
    }

    Ok(crates)
}

/// Extract crate info from a directory containing Cargo.toml.
fn crate_info_from_dir(dir: &Path) -> Result<CrateInfo> {
    let cargo_toml = dir.join("Cargo.toml");
    let content = std::fs::read_to_string(&cargo_toml)
        .with_context(|| format!("Failed to read {}", cargo_toml.display()))?;
    let parsed: toml::Value = content.parse()
        .with_context(|| format!("Failed to parse {}", cargo_toml.display()))?;

    let raw_name = parsed
        .get("package")
        .and_then(|p| p.get("name"))
        .and_then(|n| n.as_str())
        .with_context(|| format!("No [package] name in {}", cargo_toml.display()))?;

    let name = raw_name.replace('-', "_");

    let src_dir = dir.join("src");
    if !src_dir.is_dir() {
        anyhow::bail!("No src/ directory in {}", dir.display());
    }

    let src_files = collect_rs_files(&src_dir)?;

    Ok(CrateInfo {
        name,
        root: dir.to_path_buf(),
        src_dir,
        src_files,
    })
}

/// Collect all `.rs` files under a directory.
fn collect_rs_files(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    for entry in WalkDir::new(dir)
        .follow_links(false)
        .into_iter()
        .filter_entry(|e| {
            let name = e.file_name().to_string_lossy();
            !SKIP_DIRS.contains(&name.as_ref())
        })
    {
        let entry = entry?;
        if entry.file_type().is_file()
            && entry.path().extension().map(|e| e == "rs").unwrap_or(false)
        {
            files.push(entry.into_path());
        }
    }
    files.sort();
    Ok(files)
}

/// Resolve a Cargo workspace member glob pattern to actual directories.
fn resolve_glob_pattern(root: &Path, pattern: &str) -> Vec<PathBuf> {
    // Simple glob: if pattern ends with /*, list immediate subdirs.
    // Otherwise, treat as a literal path.
    if let Some(prefix) = pattern.strip_suffix("/*") {
        let base = root.join(prefix);
        if base.is_dir() {
            let mut dirs = Vec::new();
            if let Ok(entries) = std::fs::read_dir(&base) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.is_dir() && path.join("Cargo.toml").is_file() {
                        dirs.push(path);
                    }
                }
            }
            dirs.sort();
            return dirs;
        }
    }

    let path = root.join(pattern);
    if path.is_dir() {
        vec![path]
    } else {
        Vec::new()
    }
}
