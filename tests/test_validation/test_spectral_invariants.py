"""Mathematical invariant tests for spectral decomposition correctness.

These tests verify that the eigendecomposition satisfies fundamental
linear algebra properties that MUST hold if the implementation is correct.
No external baselines needed — these are self-consistency checks.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, diags

from topo_analyzer.spectral import (
    SpectralResult,
    _decompose_core,
    spectral_decomposition,
)
from topo_parser.graph import CodeGraph, Edge, EdgeKind, Node, NodeKind


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_adjacency(n: int, edges: list[tuple[int, int]]) -> csr_matrix:
    """Build a symmetric sparse adjacency matrix from edge list."""
    rows, cols, data = [], [], []
    for i, j in edges:
        rows.extend([i, j])
        cols.extend([j, i])
        data.extend([1.0, 1.0])
    return csr_matrix((data, (rows, cols)), shape=(n, n))


def _build_normalized_laplacian(adjacency: csr_matrix) -> np.ndarray:
    """Reconstruct L = I - D^{-1/2} A D^{-1/2} as a dense matrix."""
    n = adjacency.shape[0]
    degrees = np.asarray(adjacency.sum(axis=1)).flatten()
    degrees_safe = np.where(degrees > 0, degrees, 1.0)
    d_inv_sqrt = 1.0 / np.sqrt(degrees_safe)
    D_inv_sqrt = diags(d_inv_sqrt)
    laplacian = diags(np.ones(n)) - D_inv_sqrt @ adjacency @ D_inv_sqrt
    return laplacian.toarray()


def _make_code_graph(n: int, edges: list[tuple[int, int]]) -> CodeGraph:
    """Build a CodeGraph with n nodes and given edges."""
    graph = CodeGraph()
    for i in range(n):
        graph.add_node(Node(
            id=f"node_{i}",
            kind=NodeKind.FUNCTION,
            file=Path("/fake"),
            line=1,
            name=f"node_{i}",
        ))
    for i, j in edges:
        graph.add_edge(Edge(source=f"node_{i}", target=f"node_{j}", kind=EdgeKind.CALLS))
    return graph


def _complete_graph_edges(n: int) -> list[tuple[int, int]]:
    """All edges for a complete graph K_n."""
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def _path_graph_edges(n: int) -> list[tuple[int, int]]:
    """Edges for a path graph P_n: 0-1-2-..-(n-1)."""
    return [(i, i + 1) for i in range(n - 1)]


def _cycle_graph_edges(n: int) -> list[tuple[int, int]]:
    """Edges for a cycle graph C_n."""
    return _path_graph_edges(n) + [(n - 1, 0)]


def _star_graph_edges(n: int) -> list[tuple[int, int]]:
    """Edges for a star graph S_n: node 0 connected to all others."""
    return [(0, i) for i in range(1, n)]


# ---------------------------------------------------------------------------
# 1. Laplacian properties
# ---------------------------------------------------------------------------

class TestLaplacianProperties:
    """The normalized Laplacian must be symmetric and positive semidefinite."""

    def test_laplacian_is_symmetric(self):
        """L should equal its transpose."""
        edges = [(0, 1), (1, 2), (2, 3), (0, 3), (1, 3)]
        adj = _build_adjacency(4, edges)
        L = _build_normalized_laplacian(adj)
        np.testing.assert_allclose(L, L.T, atol=1e-12)

    def test_laplacian_is_symmetric_on_irregular_graph(self):
        """Symmetry holds even with highly irregular degree distribution."""
        # Star graph: node 0 has degree n-1, others have degree 1
        edges = _star_graph_edges(20)
        adj = _build_adjacency(20, edges)
        L = _build_normalized_laplacian(adj)
        np.testing.assert_allclose(L, L.T, atol=1e-12)

    def test_eigenvalues_are_nonnegative(self):
        """All eigenvalues of the normalized Laplacian must be >= 0."""
        edges = _complete_graph_edges(10)
        adj = _build_adjacency(10, edges)
        L = _build_normalized_laplacian(adj)
        eigenvalues = np.linalg.eigvalsh(L)
        assert np.all(eigenvalues >= -1e-10), f"Negative eigenvalue found: {eigenvalues.min()}"

    def test_eigenvalues_bounded_by_two(self):
        """Normalized Laplacian eigenvalues are in [0, 2]."""
        edges = _cycle_graph_edges(15)
        adj = _build_adjacency(15, edges)
        L = _build_normalized_laplacian(adj)
        eigenvalues = np.linalg.eigvalsh(L)
        assert np.all(eigenvalues >= -1e-10), f"Below 0: {eigenvalues.min()}"
        assert np.all(eigenvalues <= 2.0 + 1e-10), f"Above 2: {eigenvalues.max()}"

    def test_smallest_eigenvalue_is_zero_for_connected_graph(self):
        """A connected graph has exactly one zero eigenvalue."""
        edges = _path_graph_edges(8)
        adj = _build_adjacency(8, edges)
        L = _build_normalized_laplacian(adj)
        eigenvalues = np.sort(np.linalg.eigvalsh(L))
        assert abs(eigenvalues[0]) < 1e-10, f"Smallest eigenvalue should be ~0, got {eigenvalues[0]}"
        assert eigenvalues[1] > 1e-6, f"Second eigenvalue should be > 0, got {eigenvalues[1]}"


# ---------------------------------------------------------------------------
# 2. Eigenpair identity: L * v = lambda * v
# ---------------------------------------------------------------------------

class TestEigenpairIdentity:
    """Every returned (lambda, v) pair must satisfy L*v = lambda*v."""

    def _check_eigenpairs(self, n: int, edges: list[tuple[int, int]], k: int = 4):
        adj = _build_adjacency(n, edges)
        L = _build_normalized_laplacian(adj)
        result = _decompose_core(n, adj, k)
        assert result is not None, "Decomposition returned None"

        eigenvalues, eigenvectors = result
        for i in range(len(eigenvalues)):
            lv = L @ eigenvectors[:, i]
            lambda_v = eigenvalues[i] * eigenvectors[:, i]
            residual = np.linalg.norm(lv - lambda_v)
            assert residual < 1e-8, (
                f"Eigenpair {i} failed: ||Lv - λv|| = {residual}, λ = {eigenvalues[i]}"
            )

    def test_eigenpair_identity_path_graph(self):
        self._check_eigenpairs(10, _path_graph_edges(10))

    def test_eigenpair_identity_cycle_graph(self):
        self._check_eigenpairs(12, _cycle_graph_edges(12))

    def test_eigenpair_identity_complete_graph(self):
        self._check_eigenpairs(8, _complete_graph_edges(8))

    def test_eigenpair_identity_star_graph(self):
        self._check_eigenpairs(15, _star_graph_edges(15))

    def test_eigenpair_identity_two_cluster_graph(self):
        """Eigenpairs hold on a graph with planted community structure."""
        rng = np.random.default_rng(42)
        edges = []
        # Cluster A: nodes 0-14, cluster B: nodes 15-29
        for i in range(15):
            for j in range(i + 1, 15):
                if rng.random() < 0.3:
                    edges.append((i, j))
        for i in range(15, 30):
            for j in range(i + 1, 30):
                if rng.random() < 0.3:
                    edges.append((i, j))
        # Sparse bridge
        for i in range(15):
            for j in range(15, 30):
                if rng.random() < 0.02:
                    edges.append((i, j))
        self._check_eigenpairs(30, edges, k=6)


# ---------------------------------------------------------------------------
# 3. Eigenvector orthogonality
# ---------------------------------------------------------------------------

class TestEigenvectorOrthogonality:
    """Eigenvectors of a symmetric matrix must be orthogonal."""

    def _check_orthogonality(self, n: int, edges: list[tuple[int, int]], k: int = 4):
        adj = _build_adjacency(n, edges)
        result = _decompose_core(n, adj, k)
        assert result is not None

        _, eigenvectors = result
        # V^T V should be close to identity
        gram = eigenvectors.T @ eigenvectors
        identity = np.eye(gram.shape[0])
        np.testing.assert_allclose(gram, identity, atol=1e-8,
                                   err_msg="Eigenvectors are not orthonormal")

    def test_orthogonality_path(self):
        self._check_orthogonality(10, _path_graph_edges(10))

    def test_orthogonality_cycle(self):
        self._check_orthogonality(12, _cycle_graph_edges(12))

    def test_orthogonality_complete(self):
        self._check_orthogonality(8, _complete_graph_edges(8))

    def test_orthogonality_star(self):
        self._check_orthogonality(15, _star_graph_edges(15))


# ---------------------------------------------------------------------------
# 4. Known analytic results
# ---------------------------------------------------------------------------

class TestKnownAnalyticResults:
    """Compare against closed-form eigenvalues for standard graphs."""

    def test_complete_graph_eigenvalues(self):
        """K_n normalized Laplacian has eigenvalue 0 (once) and n/(n-1) (n-1 times)."""
        n = 10
        expected_nonzero = n / (n - 1)  # 10/9 ≈ 1.111

        adj = _build_adjacency(n, _complete_graph_edges(n))
        L = _build_normalized_laplacian(adj)
        eigenvalues = np.sort(np.linalg.eigvalsh(L))

        # First eigenvalue ≈ 0
        assert abs(eigenvalues[0]) < 1e-10
        # All others ≈ n/(n-1)
        np.testing.assert_allclose(eigenvalues[1:], expected_nonzero, atol=1e-10,
                                   err_msg=f"Expected all non-trivial eigenvalues = {expected_nonzero}")

    def test_cycle_graph_eigenvalues(self):
        """C_n normalized Laplacian eigenvalues: 1 - cos(2*pi*k/n) for k=0..n-1."""
        n = 12
        expected = sorted(1 - np.cos(2 * np.pi * k / n) for k in range(n))

        adj = _build_adjacency(n, _cycle_graph_edges(n))
        L = _build_normalized_laplacian(adj)
        eigenvalues = np.sort(np.linalg.eigvalsh(L))

        np.testing.assert_allclose(eigenvalues, expected, atol=1e-10,
                                   err_msg="Cycle graph eigenvalues don't match analytic formula")

    def test_star_graph_zero_and_one(self):
        """S_n normalized Laplacian has eigenvalue 0 (once) and 1 (n-2 times)."""
        n = 10
        adj = _build_adjacency(n, _star_graph_edges(n))
        L = _build_normalized_laplacian(adj)
        eigenvalues = np.sort(np.linalg.eigvalsh(L))

        # λ_0 = 0
        assert abs(eigenvalues[0]) < 1e-10
        # λ_1 through λ_{n-2} = 1.0
        np.testing.assert_allclose(eigenvalues[1:n - 1], 1.0, atol=1e-10,
                                   err_msg="Star graph should have n-2 eigenvalues equal to 1")
        # λ_{n-1} = 2 (star is bipartite, so max eigenvalue is 2)
        np.testing.assert_allclose(eigenvalues[-1], 2.0, atol=1e-10)

    def test_decompose_core_returns_correct_eigenvalues_for_complete(self):
        """_decompose_core should return the non-trivial eigenvalues of K_n."""
        n = 8
        expected = n / (n - 1)  # 8/7 ≈ 1.143
        adj = _build_adjacency(n, _complete_graph_edges(n))
        result = _decompose_core(n, adj, k=4)
        assert result is not None

        eigenvalues, _ = result
        # All returned eigenvalues should be n/(n-1) (trivial zero already stripped)
        np.testing.assert_allclose(eigenvalues, expected, atol=1e-6,
                                   err_msg=f"Expected eigenvalues ≈ {expected}")


# ---------------------------------------------------------------------------
# 5. Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    """Same input must produce identical output across multiple runs."""

    def test_spectral_decomposition_is_deterministic(self):
        """Running spectral_decomposition twice on the same graph gives identical results."""
        graph = _make_code_graph(20, _cycle_graph_edges(20))

        result1 = spectral_decomposition(graph, EdgeKind.CALLS, k=4)
        result2 = spectral_decomposition(graph, EdgeKind.CALLS, k=4)

        assert result1 is not None and result2 is not None
        np.testing.assert_array_equal(result1.eigenvalues, result2.eigenvalues)
        np.testing.assert_array_equal(result1.eigenvectors, result2.eigenvectors)

    def test_determinism_across_ten_runs(self):
        """Eigenvalues are bit-identical across 10 consecutive runs."""
        edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (0, 2), (1, 3)]
        graph = _make_code_graph(5, edges)

        results = [spectral_decomposition(graph, EdgeKind.CALLS, k=3) for _ in range(10)]
        assert all(r is not None for r in results)

        ref_eigenvalues = results[0].eigenvalues
        for i, r in enumerate(results[1:], 1):
            np.testing.assert_array_equal(
                r.eigenvalues, ref_eigenvalues,
                err_msg=f"Run {i} eigenvalues differ from run 0"
            )


# ---------------------------------------------------------------------------
# 6. Perturbation stability
# ---------------------------------------------------------------------------

class TestPerturbationStability:
    """Small graph changes should produce small fingerprint changes."""

    def test_adding_one_edge_causes_small_eigenvalue_shift(self):
        """Adding a single edge to a 20-node graph shifts eigenvalues slightly."""
        base_edges = _cycle_graph_edges(20)
        perturbed_edges = base_edges + [(0, 10)]  # one extra edge

        base_adj = _build_adjacency(20, base_edges)
        pert_adj = _build_adjacency(20, perturbed_edges)

        base_result = _decompose_core(20, base_adj, k=4)
        pert_result = _decompose_core(20, pert_adj, k=4)
        assert base_result is not None and pert_result is not None

        base_evals, _ = base_result
        pert_evals, _ = pert_result

        # Eigenvalue shift should be bounded — adding 1 edge to a 20-node
        # cycle (which has 20 edges) is a ~5% perturbation
        max_shift = np.max(np.abs(base_evals - pert_evals))
        assert max_shift < 0.5, f"Eigenvalue shift too large: {max_shift}"
        assert max_shift > 0, "Eigenvalues didn't change at all — edge had no effect"

    def test_eigenvalue_shift_scales_with_perturbation(self):
        """Larger perturbations cause larger eigenvalue shifts.

        We compare eigenvalues (not eigenvectors) because eigenvectors can
        flip sign or rotate within degenerate eigenspaces, making direct
        Frobenius comparison unreliable.
        """
        n = 30
        rng = np.random.default_rng(99)

        # Base graph: two clusters with some internal edges
        base_edges = []
        for i in range(15):
            for j in range(i + 1, 15):
                if rng.random() < 0.3:
                    base_edges.append((i, j))
        for i in range(15, 30):
            for j in range(i + 1, 30):
                if rng.random() < 0.3:
                    base_edges.append((i, j))
        # One bridge
        base_edges.append((7, 22))

        base_adj = _build_adjacency(n, base_edges)
        base_result = _decompose_core(n, base_adj, k=4)
        assert base_result is not None
        base_evals = base_result[0]

        # Small perturbation: add 1 cross-cluster edge
        small_adj = _build_adjacency(n, base_edges + [(3, 18)])
        small_result = _decompose_core(n, small_adj, k=4)

        # Large perturbation: add 5 cross-cluster edges
        large_adj = _build_adjacency(n, base_edges + [(3, 18), (5, 20), (8, 25), (12, 28), (1, 16)])
        large_result = _decompose_core(n, large_adj, k=4)

        assert small_result is not None and large_result is not None

        shift_small = np.sum(np.abs(small_result[0] - base_evals))
        shift_large = np.sum(np.abs(large_result[0] - base_evals))

        assert shift_large > shift_small, (
            f"Large perturbation eigenvalue shift ({shift_large:.4f}) should exceed "
            f"small perturbation shift ({shift_small:.4f})"
        )
