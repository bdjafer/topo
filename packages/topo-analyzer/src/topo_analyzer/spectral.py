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
class SpectralComponent:
    """Spectral embedding for a single connected component."""

    id: int
    node_ids: list[str]
    eigenvalues: NDArray[np.floating]
    eigenvectors: NDArray[np.floating]

    def fingerprint(self, node_id: str) -> NDArray[np.floating]:
        """Get the spectral fingerprint of a node in this component."""
        idx = self.node_ids.index(node_id)
        return self.eigenvectors[idx]


@dataclass
class SpectralResult:
    """Result of spectral decomposition."""

    total_node_ids: list[str]
    components: list[SpectralComponent]
    unassigned_components: list[list[str]]
    primary_eigenvalues: NDArray[np.floating]
    component_sizes: list[int]
    fiedler_value: float  # Second-smallest eigenvalue — algebraic connectivity

    @property
    def node_ids(self) -> list[str]:
        """Ordered node ids that received spectral fingerprints."""
        return [node_id for component in self.components for node_id in component.node_ids]

    @property
    def eigenvalues(self) -> NDArray[np.floating]:
        """Eigenvalues for the primary component used in diagnostics."""
        return self.primary_eigenvalues

    @property
    def _max_fingerprint_width(self) -> int:
        """Width to pad all fingerprints to (max eigenvector count across components)."""
        if not self.components:
            return 0
        return max(component.eigenvectors.shape[1] for component in self.components)

    @property
    def eigenvectors(self) -> NDArray[np.floating]:
        """Stack component embeddings into a single padded matrix."""
        if not self.components:
            return np.zeros((0, 0))
        width = self._max_fingerprint_width
        matrices = []
        for component in self.components:
            vectors = component.eigenvectors
            if vectors.shape[1] < width:
                vectors = np.pad(vectors, ((0, 0), (0, width - vectors.shape[1])))
            matrices.append(vectors)
        return np.vstack(matrices)

    @property
    def analyzed_node_count(self) -> int:
        """Number of nodes that received a usable embedding."""
        return len(self.node_ids)

    @property
    def unassigned_node_ids(self) -> list[str]:
        """Flattened list of nodes from components that were not clustered."""
        return [node_id for component in self.unassigned_components for node_id in component]

    @property
    def total_node_count(self) -> int:
        """Total number of nodes in the input graph."""
        return len(self.total_node_ids)

    @property
    def coverage_ratio(self) -> float:
        """Fraction of nodes that received a spectral fingerprint."""
        if self.total_node_count == 0:
            return 0.0
        return self.analyzed_node_count / self.total_node_count

    @property
    def component_count(self) -> int:
        """Number of connected components in the input graph."""
        return len(self.component_sizes)

    @property
    def clusterable_component_count(self) -> int:
        """Number of components large enough for spectral clustering."""
        return len(self.components)

    @property
    def largest_component_size(self) -> int:
        """Size of the largest connected component."""
        return max(self.component_sizes, default=0)

    @property
    def largest_component_ratio(self) -> float:
        """Fraction of nodes in the largest connected component."""
        if self.total_node_count == 0:
            return 0.0
        return self.largest_component_size / self.total_node_count

    def fingerprint(self, node_id: str) -> NDArray[np.floating]:
        """Get the spectral fingerprint of a node, padded to uniform width."""
        width = self._max_fingerprint_width
        for component in self.components:
            if node_id in component.node_ids:
                fp = component.fingerprint(node_id)
                if fp.shape[0] < width:
                    fp = np.pad(fp, (0, width - fp.shape[0]))
                return fp
        raise KeyError(f"No spectral fingerprint for node {node_id}")


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


# Default weights for multi-layer analysis, empirically validated via
# layer signal analysis (see topo_analyzer.layer_analysis):
#
# - CALLS (1.0): Strongest structural signal at all levels. At module level
#   on topo's own codebase: NMI=0.601, silhouette=0.553.
# - IMPORTS (0.5): Marginal contribution. Adding imports to calls at module
#   level slightly hurts NMI but slightly helps silhouette. Net effect ~neutral.
# - INHERITS (0.8): Rare but structurally meaningful. Cannot be empirically
#   validated on codebases without inheritance; weight set by structural reasoning.
# - CONTAINS (0.0): Captures organizational hierarchy, not coupling. At symbol
#   level, contains(1.0) inflates NMI by encoding directory structure, which
#   is circular against the directory-based NMI baseline. Kept at 0.0 per
#   design principle: "modules derived from how the code is connected, not
#   how directories are organized."
DEFAULT_LAYER_WEIGHTS: dict[EdgeKind, float] = {
    EdgeKind.CALLS: 1.0,
    EdgeKind.IMPORTS: 0.5,
    EdgeKind.CONTAINS: 0.0,
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

    For disconnected graphs, each sufficiently large connected component is
    decomposed independently. Small or edgeless components are reported as
    unassigned so they do not collapse into artificial mega-clusters.
    """
    n = adjacency.shape[0]
    n_edges = adjacency.nnz // 2  # each undirected edge stored twice
    if n < 3 or n_edges == 0:
        return None

    components = _find_connected_components(adjacency, node_ids)
    component_sizes = [len(component) for component in components]
    primary_eigenvalues = np.array([], dtype=float)
    clusterable_components: list[SpectralComponent] = []
    unassigned_components: list[list[str]] = []

    for component_id, component in enumerate(components):
        if len(component) < 3:
            unassigned_components.append([node_ids[index] for index in component])
            continue

        sub_indices = np.array(component)
        sub_adj = adjacency[np.ix_(sub_indices, sub_indices)]
        sub_n = len(component)
        sub_edges = sub_adj.nnz // 2
        if sub_edges == 0:
            unassigned_components.append([node_ids[index] for index in component])
            continue

        k_sub = min(k, sub_n - 2)
        k_sub = max(k_sub, 1)
        sub_result = _decompose_core(sub_n, sub_adj, k_sub)
        if sub_result is None:
            unassigned_components.append([node_ids[index] for index in component])
            continue

        eigenvalues, eigenvectors = sub_result
        if component_id == 0:
            primary_eigenvalues = eigenvalues
        clusterable_components.append(SpectralComponent(
            id=component_id,
            node_ids=[node_ids[index] for index in component],
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
        ))

    if not clusterable_components:
        return None

    fiedler_value = 0.0
    if len(components) == 1 and len(primary_eigenvalues) > 0:
        fiedler_value = float(primary_eigenvalues[0])

    return SpectralResult(
        total_node_ids=node_ids,
        components=clusterable_components,
        unassigned_components=unassigned_components,
        primary_eigenvalues=primary_eigenvalues,
        component_sizes=component_sizes,
        fiedler_value=fiedler_value,
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
        eigenvalues, eigenvectors = eigsh(
            laplacian,
            k=k + 1,
            which="SM",
            v0=np.linspace(1.0, 2.0, n, dtype=float),
        )
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
