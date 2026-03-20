//! WASM bindings for topo-analyzer.
//!
//! Exposes the analysis pipeline to JavaScript:
//!   analyze(jsonStr) -> jsonStr
//!   analyze_full(jsonStr) -> jsonStr

use wasm_bindgen::prelude::*;

/// Run the core structural analysis pipeline (legacy).
///
/// Takes a JSON string with {nodes, edges, k?, edge_kinds?, layer_weights?}
/// and returns a JSON string with the core AnalyzerOutput (spectral + clustering).
#[wasm_bindgen]
pub fn analyze(input_json: &str) -> Result<String, JsValue> {
    crate::analyze_json(input_json)
        .map_err(|e| JsValue::from_str(&e))
}

/// Run the complete analysis pipeline.
///
/// Takes a JSON string with {nodes, edges, k?, edge_kinds?, layer_weights?, scope?, ...}
/// and returns a JSON string matching analysis.schema.json (v3):
/// {scope, coverage, spectral, architecture, roles, issues, health}.
#[wasm_bindgen]
pub fn analyze_full(input_json: &str) -> Result<String, JsValue> {
    crate::analyze_full_json(input_json)
        .map_err(|e| JsValue::from_str(&e))
}
