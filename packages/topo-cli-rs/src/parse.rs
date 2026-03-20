//! Multi-language parser dispatch.
//!
//! Detects the project language and routes to the appropriate parser:
//! - Rust: native parser via `topo-parser-rs` crate
//! - Python: subprocess bridge invoking `python -m topo_parser`

use std::path::Path;
use std::process::Command;

use anyhow::{Context, Result, bail};

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
        Language::Rust => parse_rust_project(path),
        Language::Python => parse_python_project(path, exclude, scope),
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
fn parse_rust_project(path: &Path) -> Result<String> {
    topo_parser_rs::parse_project(path)
}

/// Parse a Python project by invoking the Python parser subprocess.
///
/// Returns the raw JSON string matching graph.schema.json.
pub fn parse_python_project(
    path: &Path,
    exclude: Option<&str>,
    scope: Option<&str>,
) -> Result<String> {
    let strategies: Vec<(&str, Vec<&str>)> = vec![
        ("uv", vec!["run", "python", "-m", "topo_parser"]),
        ("python3", vec!["-m", "topo_parser"]),
        ("python", vec!["-m", "topo_parser"]),
    ];

    let mut last_err = String::new();

    for (program, base_args) in &strategies {
        let mut cmd = Command::new(program);
        for arg in base_args {
            cmd.arg(arg);
        }
        cmd.arg(path);

        if let Some(exclude) = exclude {
            cmd.arg("--exclude").arg(exclude);
        }
        if let Some(scope) = scope {
            cmd.arg("--scope").arg(scope);
        }

        match cmd.output() {
            Ok(output) if output.status.success() => {
                let stdout = String::from_utf8(output.stdout)
                    .context("Parser output is not valid UTF-8")?;
                if stdout.trim().is_empty() {
                    bail!("Parser produced no output");
                }
                return Ok(stdout);
            }
            Ok(output) => {
                let stderr = String::from_utf8_lossy(&output.stderr);
                last_err = format!(
                    "{program}: exit {} — {}",
                    output.status,
                    stderr.trim()
                );
            }
            Err(_) => {
                continue;
            }
        }
    }

    bail!(
        "Failed to run Python parser (tried uv, python3, python).\n\
         Last error: {last_err}\n\
         Ensure topo-parser is installed: `uv sync` or `pip install topo-parser`"
    )
}
