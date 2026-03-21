//! CLI cache integration — filesystem-backed storage for parse results.
//!
//! Wraps `topo_parser::parse_project` with content-addressed caching.
//! The parser crates know nothing about caching.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use topo_cache::{CacheError, CacheStore, ContentManifest};

// ── Filesystem store ─────────────────────────────────────────────────────

/// Filesystem-backed cache store. Entries are JSON files in a directory.
pub struct FsStore {
    dir: PathBuf,
}

impl FsStore {
    pub fn new(dir: PathBuf) -> Self {
        Self { dir }
    }
}

impl CacheStore for FsStore {
    fn get(&self, key: &str) -> Option<String> {
        let path = self.dir.join(format!("{key}.json"));
        std::fs::read_to_string(&path).ok()
    }

    fn put(&mut self, key: &str, value: &str) -> Result<(), CacheError> {
        std::fs::create_dir_all(&self.dir).map_err(|e| CacheError(e.to_string()))?;
        let path = self.dir.join(format!("{key}.json"));
        let tmp = path.with_extension("tmp");
        std::fs::write(&tmp, value).map_err(|e| CacheError(e.to_string()))?;
        std::fs::rename(&tmp, &path).map_err(|e| CacheError(e.to_string()))?;
        Ok(())
    }

    fn delete(&mut self, key: &str) -> Result<(), CacheError> {
        let path = self.dir.join(format!("{key}.json"));
        let _ = std::fs::remove_file(&path);
        Ok(())
    }
}

// ── Source file walking ──────────────────────────────────────────────────

/// Build a content manifest by walking source files under `root`.
///
/// For Rust: `*.rs` + `Cargo.toml` + `Cargo.lock`
/// For Python: `*.py` + `pyproject.toml`
fn build_manifest(root: &Path, language: Option<&str>) -> Result<ContentManifest> {
    let is_python = language == Some("python")
        || (language.is_none() && !root.join("Cargo.toml").is_file());

    let extensions: &[&str] = if is_python {
        &["py"]
    } else {
        &["rs"]
    };

    let manifest_files: &[&str] = if is_python {
        &["pyproject.toml", "setup.py", "setup.cfg"]
    } else {
        &["Cargo.toml", "Cargo.lock"]
    };

    let root = std::fs::canonicalize(root).unwrap_or_else(|_| root.to_path_buf());
    let mut files: Vec<(String, Vec<u8>)> = Vec::new();

    // Walk source files.
    for entry in walkdir::WalkDir::new(&root)
        .follow_links(true)
        .into_iter()
        .filter_entry(|e| {
            let name = e.file_name().to_string_lossy();
            // Skip common non-source directories.
            !(e.file_type().is_dir()
                && matches!(
                    name.as_ref(),
                    "target" | "node_modules" | ".git" | "__pycache__" | ".venv" | "venv"
                ))
        })
    {
        let entry = match entry {
            Ok(e) => e,
            Err(_) => continue,
        };

        if !entry.file_type().is_file() {
            continue;
        }

        let path = entry.path();
        let name = entry.file_name().to_string_lossy();

        // Check if this is a manifest file (at any depth).
        let is_manifest = manifest_files.iter().any(|m| name == *m);

        // Check if this is a source file by extension.
        let is_source = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|ext| extensions.contains(&ext))
            .unwrap_or(false);

        if !is_manifest && !is_source {
            continue;
        }

        let relative = path
            .strip_prefix(&root)
            .unwrap_or(path)
            .to_string_lossy()
            .to_string();

        match std::fs::read(path) {
            Ok(content) => files.push((relative, content)),
            Err(_) => continue,
        }
    }

    Ok(ContentManifest::from_files(files.into_iter()))
}

// ── Cached parse ─────────────────────────────────────────────────────────

/// Parse with caching. Returns `(json, cache_hit)`.
///
/// On cache hit, returns the cached JSON without invoking the parser.
/// On cache miss, runs the parser and writes the result to the cache.
pub fn cached_parse(
    root: &Path,
    exclude: Option<&str>,
    scope: Option<&str>,
    language: Option<&str>,
    no_cache: bool,
) -> Result<(String, bool)> {
    if no_cache {
        let json = topo_parser::parse_project(root, exclude, scope, language)?;
        return Ok((json, false));
    }

    let cache_dir = root.join(".topo").join("cache");
    let mut store = FsStore::new(cache_dir);

    let manifest = build_manifest(root, language)
        .context("Failed to build source manifest for caching")?;
    let hash = manifest.hash();
    let version = env!("CARGO_PKG_VERSION");

    // Check cache.
    if let Some(payload) = topo_cache::load_cached(&store, "graph", &hash, version) {
        return Ok((payload, true));
    }

    // Cache miss — run the parser.
    let json = topo_parser::parse_project(root, exclude, scope, language)?;

    // Write cache (best-effort).
    let _ = topo_cache::write_cache(&mut store, "graph", &hash, version, &json);

    Ok((json, false))
}

/// Clear the parse cache for a project.
pub fn clear_cache(project_root: &Path) -> Result<()> {
    let cache_dir = project_root.join(".topo").join("cache");
    if cache_dir.exists() {
        std::fs::remove_dir_all(&cache_dir)
            .with_context(|| format!("Failed to remove {}", cache_dir.display()))?;
    }
    Ok(())
}
