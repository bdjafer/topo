//! Python parser subprocess bridge.
//!
//! Invokes `python -m topo_parser_python` to parse Python codebases.

use std::path::Path;
use std::process::Command;

use anyhow::{Context, Result, bail};

/// Parse a Python project by invoking the Python parser subprocess.
///
/// Returns the raw JSON string matching graph.schema.json.
pub fn parse_python_project(
    path: &Path,
    exclude: Option<&str>,
    scope: Option<&str>,
) -> Result<String> {
    let strategies: Vec<(&str, Vec<&str>)> = vec![
        ("uv", vec!["run", "python", "-m", "topo_parser_python"]),
        ("python3", vec!["-m", "topo_parser_python"]),
        ("python", vec!["-m", "topo_parser_python"]),
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
         Ensure topo-parser-python is installed: `uv sync` or `pip install topo-parser-python`"
    )
}
