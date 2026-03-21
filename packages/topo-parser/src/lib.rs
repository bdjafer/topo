//! Unified parser dispatch — graph types, language detection, multi-language bridge.
//!
//! Detects the project language and routes to the appropriate backend:
//! - Rust: native parser via `topo-parser-rust` (requires `rust` feature)
//! - Python: subprocess bridge invoking `python -m topo_parser_python`

pub mod graph;
pub mod python;

use std::path::Path;

use anyhow::{Result, bail};

/// Detected project language.
enum Language {
    Rust,
    Python,
}

/// Parse a project, auto-detecting language.
///
/// Returns the raw JSON string matching graph.schema.json.
pub fn parse_project(
    path: &Path,
    exclude: Option<&str>,
    scope: Option<&str>,
    language: Option<&str>,
) -> Result<String> {
    let lang = match language {
        Some("rust") => Language::Rust,
        Some("python") => Language::Python,
        Some(other) => bail!("Unknown language: {other}. Supported: rust, python"),
        None => detect_language(path),
    };

    match lang {
        Language::Rust => parse_rust(path),
        Language::Python => python::parse_python_project(path, exclude, scope),
    }
}

/// Detect language from project structure.
fn detect_language(path: &Path) -> Language {
    if path.join("Cargo.toml").is_file() {
        Language::Rust
    } else {
        Language::Python
    }
}

/// Parse a Rust project using the native parser.
#[cfg(feature = "rust")]
fn parse_rust(path: &Path) -> Result<String> {
    topo_parser_rust::parse_project(path)
}

/// Stub when Rust parser is not compiled in.
#[cfg(not(feature = "rust"))]
fn parse_rust(_path: &Path) -> Result<String> {
    bail!("Rust parsing requires the `rust` feature. Build with: cargo build --features rust")
}
