//! PyO3 bindings for topo-analyzer.
//!
//! Exposes the analysis pipeline to Python as:
//!   topo_analyzer.analyze(json_str) -> json_str
//!   topo_analyzer.analyze_full(json_str) -> json_str

use pyo3::prelude::*;

/// Run the core structural analysis pipeline (legacy).
///
/// Takes a JSON string with {nodes, edges, k?, edge_kinds?, layer_weights?}
/// and returns a JSON string with the core AnalyzerOutput (spectral + clustering).
#[pyfunction]
fn analyze(input_json: &str) -> PyResult<String> {
    crate::analyze_json(input_json)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))
}

/// Run the complete analysis pipeline.
///
/// Takes a JSON string with {nodes, edges, k?, edge_kinds?, layer_weights?, scope?, ...}
/// and returns a JSON string matching analysis.schema.json (v3):
/// {scope, coverage, spectral, architecture, roles, issues, health}.
#[pyfunction]
fn analyze_full(input_json: &str) -> PyResult<String> {
    crate::analyze_full_json(input_json)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))
}

/// Python module definition.
#[pymodule]
fn topo_analyzer(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(analyze, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_full, m)?)?;
    Ok(())
}
