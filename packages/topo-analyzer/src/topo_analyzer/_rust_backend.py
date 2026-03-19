"""
Optional Rust backend for compute-heavy analysis.

When topo_core (Rust via PyO3) is available, this module provides a fast
path for spectral decomposition, clustering, betweenness centrality, and
SCC computation — replacing scipy, numpy k-means, and hand-rolled graph
algorithms.

The Rust backend is opt-in via TOPO_BACKEND=rust environment variable.
Its primary target is WASM (where scipy/numpy are unavailable). For the
Python CLI, the scipy/numpy path remains the default to preserve existing
clustering behavior.

Falls back gracefully: if topo_core is not installed, `is_available()`
returns False and callers use the pure-Python/scipy path.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict

import numpy as np

from topo_parser.graph import CodeGraph, EdgeKind

from topo_analyzer.modules import Module, ModuleDetection
from topo_analyzer.spectral import SpectralComponent, SpectralResult

try:
    import topo_core

    _INSTALLED = True
except ImportError:
    _INSTALLED = False


def is_available() -> bool:
    """Whether the Rust backend is installed AND opted-in.

    Opt-in via TOPO_BACKEND=rust environment variable. The Rust backend
    uses a different eigendecomposition algorithm (faer dense vs scipy
    ARPACK sparse) which produces slightly different clustering. It is
    the default for WASM targets; for Python CLI the scipy path is preferred.
    """
    return _INSTALLED and os.environ.get("TOPO_BACKEND", "").lower() == "rust"


def run_core_analysis(
    graph: CodeGraph,
    *,
    edge_kind: EdgeKind = EdgeKind.CALLS,
    combined: bool = False,
    layer_weights: dict[EdgeKind, float] | None = None,
    n_modules: int | None = None,
) -> tuple[SpectralResult | None, ModuleDetection, dict[str, float], list[list[str]]]:
    """Run the full compute pipeline via Rust.

    Returns:
        (spectral_result, module_detection, betweenness, sccs)
    """
    if not _AVAILABLE:
        raise RuntimeError("topo_core is not installed")

    # Build AnalyzerInput JSON.
    node_ids = list(graph.nodes)
    nodes_json = [{"id": nid, "kind": graph.nodes[nid].kind.value} for nid in node_ids]

    edge_kinds_filter: list[str] | None = None
    weights_json: dict[str, float] | None = None

    if combined and layer_weights:
        edge_kinds_filter = [k.value for k, w in layer_weights.items() if w > 0]
        weights_json = {k.value: w for k, w in layer_weights.items() if w > 0}
    elif not combined:
        edge_kinds_filter = [edge_kind.value]

    edges_json = []
    for edge in graph.edges:
        if edge_kinds_filter and edge.kind.value not in edge_kinds_filter:
            continue
        if edge.source in graph.nodes and edge.target in graph.nodes:
            edges_json.append({
                "source": edge.source,
                "target": edge.target,
                "kind": edge.kind.value,
            })

    input_data = {
        "nodes": nodes_json,
        "edges": edges_json,
    }
    if n_modules is not None:
        input_data["k"] = n_modules
    if edge_kinds_filter:
        input_data["edge_kinds"] = edge_kinds_filter
    if weights_json:
        input_data["layer_weights"] = weights_json

    # Call Rust.
    result_json = topo_core.analyze(json.dumps(input_data))
    result = json.loads(result_json)

    # Convert to Python data structures.
    spectral = _build_spectral_result(result, node_ids)
    module_detection = _build_module_detection(result, spectral)
    betweenness: dict[str, float] = result.get("betweenness", {})
    sccs: list[list[str]] = result.get("sccs", [])

    return spectral, module_detection, betweenness, sccs


def _build_spectral_result(
    result: dict,
    total_node_ids: list[str],
) -> SpectralResult | None:
    """Reconstruct SpectralResult from Rust output."""
    fingerprints: dict[str, list[float]] = result.get("fingerprints", {})
    eigenvalues: list[float] = result.get("eigenvalues", [])
    fiedler_value: float = result.get("fiedler_value", 0.0)
    component_sizes: list[int] = result.get("component_sizes", [])
    connected_components: list[list[str]] = result.get("connected_components", [])

    if not fingerprints:
        return None

    # Reconstruct SpectralComponents from fingerprints + connected_components.
    # The Rust core decomposes per connected component — we reconstruct that.
    fp_dim = max((len(v) for v in fingerprints.values()), default=0)
    if fp_dim == 0:
        return None

    # Group nodes by connected component, preserving Rust's component order.
    node_to_component: dict[str, int] = {}
    for ci, comp_nodes in enumerate(connected_components):
        for nid in comp_nodes:
            node_to_component[nid] = ci

    # Build spectral components for components with non-zero fingerprints.
    MIN_COMPONENT_SIZE = 4
    components: list[SpectralComponent] = []
    unassigned_components: list[list[str]] = []

    for ci, comp_nodes in enumerate(connected_components):
        # Filter to nodes that have fingerprints from the Rust core.
        comp_node_ids = [nid for nid in comp_nodes if nid in fingerprints]
        if len(comp_node_ids) < MIN_COMPONENT_SIZE:
            unassigned_components.append(comp_nodes)
            continue

        # Check if all fingerprints are zero (unassigned by Rust).
        fp_matrix = np.array([fingerprints[nid] for nid in comp_node_ids])
        if np.all(fp_matrix == 0):
            unassigned_components.append(comp_nodes)
            continue

        components.append(SpectralComponent(
            id=ci,
            node_ids=comp_node_ids,
            eigenvalues=np.array(eigenvalues if ci == 0 else [], dtype=float),
            eigenvectors=fp_matrix,
        ))

    if not components:
        return None

    return SpectralResult(
        total_node_ids=total_node_ids,
        components=components,
        unassigned_components=unassigned_components,
        primary_eigenvalues=np.array(eigenvalues, dtype=float),
        component_sizes=component_sizes,
        fiedler_value=fiedler_value,
    )


def _build_module_detection(
    result: dict,
    spectral: SpectralResult | None,
) -> ModuleDetection:
    """Reconstruct ModuleDetection from Rust output."""
    clusters: dict[str, int] = result.get("clusters", {})
    silhouette: float = result.get("silhouette", 0.0)
    degenerate: bool = result.get("degenerate", False)
    component_sizes: list[int] = result.get("component_sizes", [])

    # Group nodes by cluster ID.
    cluster_members: dict[int, list[str]] = defaultdict(list)
    for nid, cid in clusters.items():
        cluster_members[cid].append(nid)

    modules: list[Module] = []
    for cid in sorted(cluster_members.keys()):
        members = sorted(cluster_members[cid])
        modules.append(Module(
            id=cid,
            node_ids=members,
            confidence=0.5 if degenerate else max(0.0, min(1.0, silhouette)),
        ))

    clustered_count = sum(m.size for m in modules)
    unassigned_count = 0
    if spectral:
        unassigned_count = len(spectral.unassigned_node_ids)

    return ModuleDetection(
        modules=modules,
        chosen_k=len(modules) if modules else None,
        silhouette=silhouette if not degenerate else None,
        component_count=len(component_sizes),
        clustered_node_count=clustered_count,
        unassigned_node_count=unassigned_count,
        package_fallback=degenerate,
    )
