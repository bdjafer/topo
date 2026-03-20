"""
Rust backend bridge for structural analysis.

Sends the raw graph plus projection config to the topo-analyzer Rust extension
(via PyO3) and returns schema-compliant JSON. The Rust side handles projection,
spectral decomposition, clustering, module enrichment, role classification,
anomaly detection, finding synthesis, and serialization.
"""

from __future__ import annotations

import json
import os

from topo_parser.graph import CodeGraph

try:
    import topo_analyzer

    _INSTALLED = True
except ImportError:
    _INSTALLED = False


def is_available() -> bool:
    """Whether the Rust backend is installed AND opted-in."""
    return _INSTALLED and os.environ.get("TOPO_BACKEND", "").lower() == "rust"


def is_full_available() -> bool:
    """Whether the Rust full analysis pipeline (analyze_full) is available."""
    return is_available() and hasattr(topo_analyzer, "analyze_full")


def run_full_analysis(
    graph: CodeGraph,
    *,
    projection_config: "AnalysisProjectionConfig",
    n_modules: int | None = None,
) -> dict:
    """Run the complete analysis pipeline via Rust, returning schema-compliant JSON.

    Sends the RAW graph to Rust along with projection config. Rust handles
    projection, spectral analysis, clustering, module enrichment, role
    classification, anomaly detection, finding synthesis, and serialization.

    Returns a dict matching analysis.schema.json (v3).
    """
    if not is_full_available():
        raise RuntimeError("topo_analyzer.analyze_full is not available")

    # Serialize the raw graph — no Python projection needed.
    nodes_json = [
        {
            "id": nid,
            "kind": node.kind.value,
            "file": node.file,
            "line": node.line,
        }
        for nid, node in graph.nodes.items()
    ]
    edges_json = [
        {
            "source": edge.source,
            "target": edge.target,
            "kind": edge.kind.value,
        }
        for edge in graph.edges
    ]

    # Build layer weights if present.
    weights_json = None
    if projection_config.layer_weights:
        weights_json = {
            k.value: w for k, w in projection_config.layer_weights.items() if w > 0
        }

    edge_kinds = [k.value for k in projection_config.edge_kinds]

    input_data: dict = {
        "nodes": nodes_json,
        "edges": edges_json,
        "edge_kinds": edge_kinds,
        "projection": {
            "level": projection_config.level.value,
            "source_node_kinds": [k.value for k in projection_config.source_node_kinds],
            "edge_kinds": edge_kinds,
            "scope_roots": [str(r) for r in projection_config.scope_roots],
            "internal_only": projection_config.internal_only,
        },
    }
    if n_modules is not None:
        input_data["k"] = n_modules
    if weights_json:
        input_data["layer_weights"] = weights_json

    result_json = topo_analyzer.analyze_full(json.dumps(input_data))
    return json.loads(result_json)
