//! Export R-GIN training features as NPZ + JSON metadata sidecar.

use std::collections::HashMap;
use std::fs::File;

use anyhow::{Result, bail};
use ndarray::Array2;
use ndarray_npy::NpzWriter;

use crate::args::{Cli, ExportFeaturesArgs};

/// Export all R-GIN training features to an NPZ file with a JSON metadata sidecar.
///
/// Features exported:
/// - `semantic`: float32[n, 768] — semantic embeddings (zeros if unavailable)
/// - `spectral_vecs`: float32[n, 16] — Laplacian eigenvector components
/// - `spectral_vals`: float32[n, 16] — Laplacian eigenvalues per node's component
/// - `rwpe`: float32[n, 16] — random walk positional encodings
/// - `tree_features`: int32[n, 4] — containment-tree features
/// - `node_types`: int32[n] — canonical node type indices
/// - `edge_index_{calls,imports,inherits}`: int32[2, m] — typed edge indices
pub fn cmd_export_features(args: &ExportFeaturesArgs, cli: &Cli) -> Result<()> {
    // ── 1. Load graph JSON ──────────────────────────────────────────────
    let graph_json = if let Some(ref path) = args.path {
        let (json, _cache_hit) = crate::cache::cached_parse(path, None, None, None, cli.no_cache)?;
        json
    } else if let Some(ref input_path) = args.input {
        std::fs::read_to_string(input_path)
            .map_err(|e| anyhow::anyhow!("Failed to read {}: {e}", input_path.display()))?
    } else {
        bail!("Either <path> or --input must be provided");
    };

    // ── 2. Build AnalyzerInput ──────────────────────────────────────────
    // edge_kinds = None so that defines edges are included (needed for tree features).
    let parsed: serde_json::Value = serde_json::from_str(&graph_json)?;
    let nodes = parsed
        .get("nodes")
        .cloned()
        .unwrap_or(serde_json::Value::Array(vec![]));
    let edges = parsed
        .get("edges")
        .cloned()
        .unwrap_or(serde_json::Value::Array(vec![]));

    let analyzer_input_json = serde_json::json!({
        "nodes": nodes,
        "edges": edges,
    });
    let input: topo_analyzer::types::AnalyzerInput =
        serde_json::from_value(analyzer_input_json)?;

    // ── 3. Build graph and run spectral analysis ────────────────────────
    let graph = topo_analyzer::graph::Graph::from_input(&input);

    if graph.n == 0 {
        eprintln!("Warning: graph has 0 nodes, writing empty NPZ");
    }

    let spectral = topo_analyzer::spectral::decompose(&graph, 16);

    // ── 4. Compute all features ─────────────────────────────────────────
    let (spectral_vecs, spectral_vals) =
        topo_analyzer::spectral::spectral_pe_export(&spectral, graph.n, 16);

    let rwpe = topo_analyzer::rwpe::compute_rwpe(&graph, 16, 512);

    let tree_features = topo_analyzer::tree::compute_tree_features(&graph);

    let node_types: Vec<usize> = graph
        .node_kinds
        .iter()
        .map(|k| topo_analyzer::types::node_type_index(k))
        .collect();

    // ── 5. Load semantic embeddings ─────────────────────────────────────
    let embed_dim = 768;
    let mut semantic = vec![vec![0.0f32; embed_dim]; graph.n];
    if let Some(ref emb_path) = cli.embeddings {
        let emb_json = std::fs::read_to_string(emb_path)
            .map_err(|e| anyhow::anyhow!("Failed to read embeddings: {e}"))?;
        let emb_data: HashMap<String, Vec<f32>> = serde_json::from_str(&emb_json)?;
        for (i, node_id) in graph.node_ids.iter().enumerate() {
            if let Some(vec) = emb_data.get(node_id).filter(|v| v.len() == embed_dim) {
                semantic[i].clone_from(vec);
            }
        }
    }

    // ── 6. Write NPZ file ───────────────────────────────────────────────
    let file = File::create(&args.output)
        .map_err(|e| anyhow::anyhow!("Failed to create {}: {e}", args.output.display()))?;
    let mut npz = NpzWriter::new(file);

    // semantic: float32[n, 768]
    let semantic_arr =
        Array2::from_shape_fn((graph.n, embed_dim), |(i, j)| semantic[i][j]);
    npz.add_array("semantic", &semantic_arr)?;

    // spectral_vecs: float32[n, 16]
    let spec_vecs_arr =
        Array2::from_shape_fn((graph.n, 16), |(i, j)| spectral_vecs[i][j] as f32);
    npz.add_array("spectral_vecs", &spec_vecs_arr)?;

    // spectral_vals: float32[n, 16]
    let spec_vals_arr =
        Array2::from_shape_fn((graph.n, 16), |(i, j)| spectral_vals[i][j] as f32);
    npz.add_array("spectral_vals", &spec_vals_arr)?;

    // rwpe: float32[n, 16]
    let rwpe_arr =
        Array2::from_shape_fn((graph.n, 16), |(i, j)| rwpe[i][j] as f32);
    npz.add_array("rwpe", &rwpe_arr)?;

    // tree_features: int32[n, 4]
    let tree_arr =
        Array2::from_shape_fn((graph.n, 4), |(i, j)| tree_features[i][j] as i32);
    npz.add_array("tree_features", &tree_arr)?;

    // node_types: int32[n]
    let types_arr = ndarray::Array1::from_shape_fn(graph.n, |i| node_types[i] as i32);
    npz.add_array("node_types", &types_arr)?;

    // Edge indices per type: int32[2, m]
    for edge_kind in &["calls", "imports", "inherits"] {
        let edges = graph.edges_of_kind(edge_kind);
        let m = edges.len();
        let edge_arr = Array2::from_shape_fn((2, m), |(row, col)| {
            if row == 0 {
                edges[col].0 as i32
            } else {
                edges[col].1 as i32
            }
        });
        npz.add_array(format!("edge_index_{edge_kind}"), &edge_arr)?;
    }

    npz.finish()?;

    // ── 7. Write metadata JSON sidecar ──────────────────────────────────
    let meta_path = args.output.with_extension("meta.json");

    let edge_counts = serde_json::json!({
        "calls": graph.edges_of_kind("calls").len(),
        "imports": graph.edges_of_kind("imports").len(),
        "inherits": graph.edges_of_kind("inherits").len(),
    });

    let metadata = serde_json::json!({
        "n_nodes": graph.n,
        "n_edges": edge_counts,
        "node_ids": graph.node_ids,
        "n_components": spectral.component_sizes.len(),
        "fiedler_value": spectral.fiedler_value,
    });

    std::fs::write(&meta_path, serde_json::to_string_pretty(&metadata)?)
        .map_err(|e| anyhow::anyhow!("Failed to write {}: {e}", meta_path.display()))?;

    // ── 8. Print summary ────────────────────────────────────────────────
    eprintln!("Exported {} nodes to {}", graph.n, args.output.display());
    eprintln!("Metadata: {}", meta_path.display());

    Ok(())
}
