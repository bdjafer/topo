//! PyO3 bindings for topo-formatter.
//!
//! Exposes the text formatter to Python as:
//!   topo_formatter.format_text(json_str, ...) -> str

use std::collections::HashMap;

use pyo3::prelude::*;

/// Format analysis JSON into human-readable text.
///
/// Takes a JSON string (analysis.schema.json v3) and returns formatted text.
#[pyfunction]
#[pyo3(signature = (data_json, verbose=false, diagnostics=false, ignores_json=None, project_root=None, color=false))]
fn format_text(
    data_json: &str,
    verbose: bool,
    diagnostics: bool,
    ignores_json: Option<&str>,
    project_root: Option<&str>,
    color: bool,
) -> PyResult<String> {
    let data: serde_json::Value = serde_json::from_str(data_json)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let ignores: HashMap<String, String> = match ignores_json {
        Some(s) => serde_json::from_str(s)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?,
        None => HashMap::new(),
    };
    let root = project_root.map(std::path::Path::new);
    Ok(crate::format_text(&data, verbose, diagnostics, &ignores, root, color))
}

/// Python module definition.
#[pymodule]
fn topo_formatter(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(format_text, m)?)?;
    Ok(())
}
