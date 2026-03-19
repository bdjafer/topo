//! WASM bindings for topo-core.
//!
//! Exposes the core analysis pipeline to JavaScript:
//!   analyze(jsonStr) -> jsonStr

use wasm_bindgen::prelude::*;

/// Run the full structural analysis pipeline.
///
/// Takes a JSON string with {nodes, edges, k?, edge_kinds?, layer_weights?}
/// and returns a JSON string with the full AnalyzerOutput.
#[wasm_bindgen]
pub fn analyze(input_json: &str) -> Result<String, JsValue> {
    crate::analyze_json(input_json)
        .map_err(|e| JsValue::from_str(&e))
}
