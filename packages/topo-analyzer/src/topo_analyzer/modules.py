"""
Module detection via spectral clustering.

Uses spectral fingerprints to identify structurally cohesive clusters
of code entities — the actual architectural modules, derived from
how the code is connected rather than how directories are organized.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from topo_analyzer.spectral import SpectralResult


@dataclass
class Module:
    """A detected structural module (cluster of tightly coupled entities)."""

    id: int
    node_ids: list[str]

    @property
    def size(self) -> int:
        return len(self.node_ids)


def detect_modules(spectral: SpectralResult, n_modules: int | None = None) -> list[Module]:
    """
    Detect structural modules from spectral fingerprints using k-means clustering.

    Args:
        spectral: Result from spectral decomposition.
        n_modules: Number of modules to detect. If None, estimated from eigenvalue gaps.

    Returns:
        List of detected modules.
    """
    if n_modules is None:
        n_modules = _estimate_k(spectral.eigenvalues)

    # Simple k-means on the spectral fingerprints
    labels = _kmeans(spectral.eigenvectors, n_modules)

    modules = []
    for i in range(n_modules):
        member_ids = [
            spectral.node_ids[j]
            for j in range(len(spectral.node_ids))
            if labels[j] == i
        ]
        if member_ids:
            modules.append(Module(id=i, node_ids=member_ids))

    return modules


def _estimate_k(eigenvalues: np.ndarray) -> int:
    """Estimate number of clusters from eigenvalue gaps (eigengap heuristic)."""
    if len(eigenvalues) < 2:
        return 2
    gaps = np.diff(eigenvalues)
    # The largest gap suggests the number of well-separated clusters
    k = int(np.argmax(gaps)) + 2  # +2 because gap at index i means i+2 clusters
    return max(2, min(k, len(eigenvalues)))


def _kmeans(X: np.ndarray, k: int, max_iter: int = 100) -> np.ndarray:
    """Minimal k-means implementation. No sklearn dependency needed for this."""
    n = X.shape[0]
    rng = np.random.default_rng(42)
    # k-means++ initialization
    centroids = [X[rng.integers(n)]]
    for _ in range(k - 1):
        dists = np.min([np.sum((X - c) ** 2, axis=1) for c in centroids], axis=0)
        probs = dists / dists.sum()
        centroids.append(X[rng.choice(n, p=probs)])
    centroids_arr = np.array(centroids)

    labels = np.zeros(n, dtype=int)
    for _ in range(max_iter):
        # Assign
        dists = np.array([np.sum((X - c) ** 2, axis=1) for c in centroids_arr])
        new_labels = np.argmin(dists, axis=0)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        # Update
        for i in range(k):
            mask = labels == i
            if mask.any():
                centroids_arr[i] = X[mask].mean(axis=0)

    return labels
