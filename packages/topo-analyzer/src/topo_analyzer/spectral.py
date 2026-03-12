"""
Spectral decomposition of code graphs.

Computes eigendecomposition of the graph Laplacian to extract
global structural properties. Each node gets a spectral fingerprint —
its coordinates in the eigenspace — which encodes its structural position
in the graph.

Uses sparse matrices throughout so memory scales with the number of edges
rather than O(n^2) with the number of nodes.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigsh

from topo_parser.graph import CodeGraph, EdgeKind

try:
    from scipy.sparse.csgraph import connected_components as _scipy_cc

    _HAS_SCIPY_CC = True
except ImportError:
    _HAS_SCIPY_CC = False


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


def _find_connected_components(
    adj_matrix: csr_matrix, node_ids: list[str]
) -> list[list[int]]:
    """
    Find connected components in an adjacency structure.

    Uses scipy.sparse.csgraph.connected_components when available,
    otherwise falls back to a simple BFS.

    Args:
        adj_matrix: Sparse adjacency matrix (n x n).
        node_ids: Ordered list of node identifiers (length n).

    Returns:
        List of components, each a list of node indices. Components are
        sorted by size (largest first).
    """
    n = len(node_ids)

    if _HAS_SCIPY_CC:
        n_components, labels = _scipy_cc(adj_matrix, directed=False)
        components: dict[int, list[int]] = {}
        for idx, label in enumerate(labels):
            components.setdefault(label, []).append(idx)
        result = list(components.values())
    else:
        # BFS fallback
        visited = [False] * n
        result = []
        coo = adj_matrix.tocoo()
        adj: dict[int, list[int]] = {i: [] for i in range(n)}
        for r, c in zip(coo.row, coo.col):
            adj[r].append(c)

        for start in range(n):
            if visited[start]:
                continue
            component: list[int] = []
            queue = deque([start])
            visited[start] = True
            while queue:
                node = queue.popleft()
                component.append(node)
                for neighbor in adj[node]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append(neighbor)
            result.append(component)

    # Sort components by size, largest first
    result.sort(key=len, reverse=True)
    return result


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
    node_ids = list(graph.nodes)
    node_index = {nid: i for i, nid in enumerate(node_ids)}
    n = len(node_ids)

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    for edge in graph.edges_by_kind(edge_kind):
        if edge.source in node_index and edge.target in node_index:
            i, j = node_index[edge.source], node_index[edge.target]
            rows.extend([i, j])
            cols.extend([j, i])
            data.extend([1.0, 1.0])

    adjacency = csr_matrix((data, (rows, cols)), shape=(n, n))

    return _decompose(node_ids, adjacency, k)


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

    node_ids = list(graph.nodes)
    node_index = {nid: i for i, nid in enumerate(node_ids)}
    n = len(node_ids)

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    for edge_kind, weight in layer_weights.items():
        if weight <= 0:
            continue
        for edge in graph.edges_by_kind(edge_kind):
            if edge.source in node_index and edge.target in node_index:
                i, j = node_index[edge.source], node_index[edge.target]
                rows.extend([i, j])
                cols.extend([j, i])
                data.extend([weight, weight])

    # Duplicate entries at the same (i,j) are summed by csr_matrix,
    # which naturally handles multi-layer weight accumulation.
    adjacency = csr_matrix((data, (rows, cols)), shape=(n, n))

    return _decompose(node_ids, adjacency, k)


def _decompose(
    node_ids: list[str],
    adjacency: csr_matrix,
    k: int,
) -> SpectralResult | None:
    """
    Run eigendecomposition on a sparse adjacency matrix.

    Builds the normalized Laplacian L = I - D^{-1/2} A D^{-1/2} as a
    sparse matrix, then computes the k+1 smallest eigenvalues via
    scipy.sparse.linalg.eigsh.

    For disconnected graphs, spectral decomposition is performed on the
    largest connected component only. Nodes outside the largest component
    get zero-valued eigenvector entries, and fiedler_value is set to 0.0.
    """
    n = adjacency.shape[0]
    n_edges = adjacency.nnz // 2  # each undirected edge stored twice
    if n < 3 or n_edges == 0:
        return None

    # Detect connected components
    components = _find_connected_components(adjacency, node_ids)
    is_disconnected = len(components) > 1

    if is_disconnected:
        # Work on the largest connected component only
        largest = components[0]
        if len(largest) < 3:
            return None
        sub_indices = np.array(largest)
        sub_adj = adjacency[np.ix_(sub_indices, sub_indices)]
        sub_n = len(largest)
        k_sub = min(k, sub_n - 2)
        k_sub = max(k_sub, 1)
        sub_result = _decompose_core(sub_n, sub_adj, k_sub)
        if sub_result is None:
            return None
        sub_eigenvalues, sub_eigenvectors = sub_result

        # Embed sub-component results back into full-size arrays.
        # Nodes outside the largest component get zero eigenvector entries.
        full_eigenvectors = np.zeros((n, sub_eigenvectors.shape[1]), dtype=sub_eigenvectors.dtype)
        for local_idx, global_idx in enumerate(largest):
            full_eigenvectors[global_idx] = sub_eigenvectors[local_idx]

        return SpectralResult(
            node_ids=node_ids,
            eigenvalues=sub_eigenvalues,
            eigenvectors=full_eigenvectors,
            fiedler_value=0.0,
        )
    else:
        # Graph is fully connected — proceed normally
        if n < k + 1:
            k = max(n - 2, 1)
        result = _decompose_core(n, adjacency, k)
        if result is None:
            return None
        eigenvalues, eigenvectors = result
        return SpectralResult(
            node_ids=node_ids,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            fiedler_value=float(eigenvalues[0]),
        )


def _decompose_core(
    n: int,
    adjacency: csr_matrix,
    k: int,
) -> tuple[NDArray[np.floating], NDArray[np.floating]] | None:
    """
    Core eigendecomposition on a (connected) adjacency matrix.

    Returns (eigenvalues, eigenvectors) with the trivial zero eigenvalue
    already stripped, or None on failure.
    """
    # Degree vector from the adjacency matrix
    degrees = np.asarray(adjacency.sum(axis=1)).flatten()

    # Guard against isolated nodes (degree 0) — use 1 to avoid division by zero
    degrees_safe = np.where(degrees > 0, degrees, 1.0)
    d_inv_sqrt = 1.0 / np.sqrt(degrees_safe)

    # Normalized Laplacian: L = I - D^{-1/2} A D^{-1/2}
    D_inv_sqrt = diags(d_inv_sqrt)
    laplacian = diags(np.ones(n)) - D_inv_sqrt @ adjacency @ D_inv_sqrt

    # k+1 smallest eigenvalues (skip the trivial zero eigenvalue)
    try:
        eigenvalues, eigenvectors = eigsh(laplacian, k=k + 1, which="SM")
    except Exception:
        # eigsh can fail on very small graphs;
        # fall back to dense decomposition as a last resort.
        try:
            dense_lap = laplacian.toarray()
            all_eigenvalues, all_eigenvectors = np.linalg.eigh(dense_lap)
            eigenvalues = all_eigenvalues[: k + 1]
            eigenvectors = all_eigenvectors[:, : k + 1]
        except Exception:
            return None

    # Sort by eigenvalue and skip the first (trivial) one
    order = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[order][1:]  # skip lambda_0 ~ 0
    eigenvectors = eigenvectors[:, order][:, 1:]

    return eigenvalues, eigenvectors
