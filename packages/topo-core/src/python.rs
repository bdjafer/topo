//! PyO3 bindings for topo-core.
//!
//! Exposes the core analysis pipeline to Python as a single function:
//!   topo_core.analyze(json_str) -> json_str

use pyo3::prelude::*;

/// Run the full structural analysis pipeline.
///
/// Takes a JSON string with {nodes, edges, k?, edge_kinds?, layer_weights?}
/// and returns a JSON string with the full AnalyzerOutput.
#[pyfunction]
fn analyze(input_json: &str) -> PyResult<String> {
    crate::analyze_json(input_json)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))
}

/// Python module definition.
#[pymodule]
fn topo_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(analyze, m)?)?;
    Ok(())
}
