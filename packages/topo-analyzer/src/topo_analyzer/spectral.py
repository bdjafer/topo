"""
Spectral decomposition of code graphs.

Computes eigendecomposition of the graph Laplacian to extract
global structural properties. Each node gets a spectral fingerprint —
its coordinates in the eigenspace — which encodes its structural position
in the graph.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
from numpy.typing import NDArray
from scipy.sparse.linalg import eigsh

from topo_parser.graph import CodeGraph, EdgeKind


@dataclass
class SpectralResult:
    """Result of spectral decomposition."""

    node_ids: list[str]  # Ordered list of node ids, corresponding to matrix rows
    eigenvalues: NDArray[np.floating]  # k smallest non-trivial eigenvalues
    eigenvectors: NDArray[np.floating]  # (n_nodes, k) matrix — spectral fingerprints
    fiedler_value: float  # Second-smallest eigenvalue — algebraic connectivity

    def fingerprint(self, node_id: str) -> NDArray[np.floating]:
        """Get the spectral fingerprint of a node."""
        idx = self.node_ids.index(node_id)
        return self.eigenvectors[idx]


def spectral_decomposition(
    graph: CodeGraph,
    edge_kind: EdgeKind = EdgeKind.CALLS,
    k: int = 8,
) -> SpectralResult | None:
    """
    Compute spectral decomposition of a single graph layer.

    Args:
        graph: The code graph to analyze.
        edge_kind: Which relationship layer to decompose.
        k: Number of eigenvectors to compute.

    Returns:
        SpectralResult with eigenvalues and fingerprints, or None if
        the graph is too small for meaningful decomposition.
    """
    # Build a NetworkX graph from the specified layer
    nx_graph = nx.Graph()
    for node_id in graph.nodes:
        nx_graph.add_node(node_id)

    for edge in graph.edges_by_kind(edge_kind):
        if edge.source in graph.nodes and edge.target in graph.nodes:
            nx_graph.add_edge(edge.source, edge.target)

    return _decompose(nx_graph, k)


# Default weights for multi-layer analysis. Calls are the strongest
# signal, imports and containment provide context, inheritance is rare
# but structurally meaningful.
DEFAULT_LAYER_WEIGHTS: dict[EdgeKind, float] = {
    EdgeKind.CALLS: 1.0,
    EdgeKind.IMPORTS: 0.5,
    EdgeKind.CONTAINS: 0.3,
    EdgeKind.INHERITS: 0.8,
}


def spectral_decomposition_multilayer(
    graph: CodeGraph,
    layer_weights: dict[EdgeKind, float] | None = None,
    k: int = 8,
) -> SpectralResult | None:
    """
    Compute spectral decomposition over multiple graph layers combined.

    Builds a single weighted adjacency matrix by summing contributions
    from each layer scaled by its weight. This gives every node a
    structural fingerprint that reflects calls, imports, containment,
    and inheritance simultaneously.

    Args:
        graph: The code graph to analyze.
        layer_weights: Weight per EdgeKind. Defaults to DEFAULT_LAYER_WEIGHTS.
        k: Number of eigenvectors to compute.

    Returns:
        SpectralResult, or None if the graph is too small.
    """
    if layer_weights is None:
        layer_weights = DEFAULT_LAYER_WEIGHTS

    nx_graph = nx.Graph()
    for node_id in graph.nodes:
        nx_graph.add_node(node_id)

    for edge_kind, weight in layer_weights.items():
        if weight <= 0:
            continue
        for edge in graph.edges_by_kind(edge_kind):
            if edge.source in graph.nodes and edge.target in graph.nodes:
                src, tgt = edge.source, edge.target
                if nx_graph.has_edge(src, tgt):
                    nx_graph[src][tgt]["weight"] += weight
                else:
                    nx_graph.add_edge(src, tgt, weight=weight)

    return _decompose(nx_graph, k)


def _decompose(nx_graph: nx.Graph, k: int) -> SpectralResult | None:
    """Run eigendecomposition on a NetworkX graph."""
    n = nx_graph.number_of_nodes()
    n_edges = nx_graph.number_of_edges()
    if n < 3 or n_edges == 0:
        return None
    if n < k + 1:
        k = max(n - 2, 1)

    # Compute normalized Laplacian eigenvectors
    node_ids = list(nx_graph.nodes())
    laplacian = nx.normalized_laplacian_matrix(nx_graph).astype(float)

    # k+1 smallest eigenvalues (skip the trivial zero eigenvalue)
    eigenvalues, eigenvectors = eigsh(laplacian, k=k + 1, which="SM")

    # Sort by eigenvalue and skip the first (trivial) one
    order = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[order][1:]  # skip λ₀ ≈ 0
    eigenvectors = eigenvectors[:, order][:, 1:]

    return SpectralResult(
        node_ids=node_ids,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        fiedler_value=float(eigenvalues[0]),
    )
