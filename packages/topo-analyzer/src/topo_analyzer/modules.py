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
        n_modules: Number of modules to detect. If None, selected by best
                   silhouette score across k=2..max_k.

    Returns:
        List of detected modules.
    """
    if n_modules is None:
        n_modules = _estimate_k(spectral.eigenvectors)

    labels = _kmeans(spectral.eigenvectors, n_modules)
    return _labels_to_modules(labels, spectral.node_ids, n_modules)


def _labels_to_modules(labels: np.ndarray, node_ids: list[str], k: int) -> list[Module]:
    """Convert label array to Module list."""
    modules = []
    for i in range(k):
        member_ids = [node_ids[j] for j in range(len(node_ids)) if labels[j] == i]
        if member_ids:
            modules.append(Module(id=i, node_ids=member_ids))
    return modules


def _estimate_k(X: np.ndarray, max_k: int = 8) -> int:
    """Select k using silhouette scores with diminishing-returns cutoff.

    Tries k=2..max_k. Computes silhouette gains between consecutive k values.
    Picks the first k where the subsequent gain drops below the mean gain,
    balancing cluster quality against over-fragmentation.
    """
    max_k = min(max_k, X.shape[0] - 1)
    if max_k < 2:
        return 2

    scores = []
    for k in range(2, max_k + 1):
        labels = _kmeans(X, k)
        scores.append(_silhouette_score(X, labels))

    if len(scores) < 3:
        return int(np.argmax(scores)) + 2

    gains = [scores[i + 1] - scores[i] for i in range(len(scores) - 1)]
    mean_gain = np.mean([g for g in gains if g > 0]) if any(g > 0 for g in gains) else 0

    # Pick first k where subsequent gain drops below mean gain
    for i in range(len(gains) - 1):
        if gains[i] > 0 and gains[i + 1] < mean_gain:
            return i + 3  # k = i+2 was good, next gain dropped, so use i+3

    # No clear elbow — pick k with best score
    return int(np.argmax(scores)) + 2


def _silhouette_score(X: np.ndarray, labels: np.ndarray, max_sample: int = 5000) -> float:
    """Compute mean silhouette score for a clustering.

    For each sampled point, silhouette = (b - a) / max(a, b) where:
    - a = mean distance to points in same cluster
    - b = mean distance to points in nearest other cluster

    For large datasets (n > max_sample), uses a random sample to avoid
    O(n^2) memory. Uses centroid-based approximation for b when clusters
    are large.
    """
    n = len(labels)
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return -1.0

    # Precompute cluster centroids and sizes
    centroids = {}
    for label in unique_labels:
        mask = labels == label
        centroids[label] = X[mask].mean(axis=0)

    # Sample if too large
    rng = np.random.default_rng(42)
    if n > max_sample:
        sample_idx = rng.choice(n, max_sample, replace=False)
    else:
        sample_idx = np.arange(n)

    silhouettes = np.zeros(len(sample_idx))
    for si, i in enumerate(sample_idx):
        my_label = labels[i]
        same = labels == my_label
        same_count = same.sum() - 1  # exclude self
        if same_count <= 0:
            silhouettes[si] = 0.0
            continue

        # a = mean distance to same-cluster centroid (fast approximation)
        a = np.linalg.norm(X[i] - centroids[my_label])

        # b = distance to nearest other cluster centroid
        b = np.inf
        for label in unique_labels:
            if label == my_label:
                continue
            dist = np.linalg.norm(X[i] - centroids[label])
            if dist < b:
                b = dist

        silhouettes[si] = (b - a) / max(a, b) if max(a, b) > 0 else 0.0

    return float(silhouettes.mean())


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
