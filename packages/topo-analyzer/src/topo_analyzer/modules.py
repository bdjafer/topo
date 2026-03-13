"""
Module detection via spectral clustering.

Uses spectral fingerprints to identify structurally cohesive clusters
of code entities — the actual architectural modules, derived from
how the code is connected rather than how directories are organized.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from topo_analyzer.spectral import SpectralComponent, SpectralResult


@dataclass
class Module:
    """A detected structural module (cluster of tightly coupled entities)."""

    id: int
    node_ids: list[str]
    component_id: int | None = None
    cohesion: float | None = None
    separation: float | None = None
    confidence: float = 0.0
    unassigned: bool = False

    @property
    def size(self) -> int:
        return len(self.node_ids)


@dataclass
class ModuleDetection:
    """Detected structural modules plus clustering diagnostics."""

    modules: list[Module]
    chosen_k: int | None
    silhouette: float | None
    component_count: int
    clustered_node_count: int
    unassigned_node_count: int
    package_fallback: bool = False


def detect_modules(
    spectral: SpectralResult,
    n_modules: int | None = None,
) -> ModuleDetection:
    """
    Detect structural modules from spectral fingerprints using k-means clustering.

    Args:
        spectral: Result from spectral decomposition.
        n_modules: Number of modules to detect. If None, selected by best
                   silhouette score across k=2..max_k.

    Returns:
        Detected modules plus clustering diagnostics.
    """
    modules: list[Module] = []
    next_module_id = 0
    chosen_ks: list[int] = []
    silhouettes: list[tuple[float, int]] = []

    for component in spectral.components:
        explicit_k = n_modules if len(spectral.components) == 1 else None
        component_modules, component_k, component_silhouette = _cluster_component(
            component,
            next_module_id,
            explicit_k,
        )
        modules.extend(component_modules)
        next_module_id += len(component_modules)
        chosen_ks.append(component_k)
        if component_silhouette is not None:
            silhouettes.append((component_silhouette, len(component.node_ids)))

    for unassigned_nodes in spectral.unassigned_components:
        modules.append(Module(
            id=next_module_id,
            node_ids=unassigned_nodes,
            confidence=0.0,
            unassigned=True,
        ))
        next_module_id += 1

    chosen_k = sum(chosen_ks) if chosen_ks else None
    silhouette = None
    if silhouettes:
        total_weight = sum(weight for _, weight in silhouettes)
        silhouette = sum(score * weight for score, weight in silhouettes) / total_weight

    # When k was auto-detected and spectral clustering is degenerate,
    # fall back to package-based grouping which uses the structure
    # already encoded in the node IDs.
    used_fallback = False
    if n_modules is None and _is_degenerate(modules, silhouette):
        all_node_ids = [nid for m in modules for nid in m.node_ids]
        modules = _package_grouping(all_node_ids)
        chosen_k = len(modules)
        silhouette = None
        used_fallback = True

    return ModuleDetection(
        modules=modules,
        chosen_k=chosen_k,
        silhouette=silhouette,
        component_count=spectral.component_count,
        clustered_node_count=spectral.analyzed_node_count,
        unassigned_node_count=len(spectral.unassigned_node_ids),
        package_fallback=used_fallback,
    )


def _labels_to_modules(
    labels: np.ndarray,
    node_ids: list[str],
    k: int,
    *,
    start_id: int,
    component_id: int,
) -> list[Module]:
    """Convert label array to Module list."""
    modules = []
    for i in range(k):
        member_ids = [node_ids[j] for j in range(len(node_ids)) if labels[j] == i]
        if member_ids:
            modules.append(Module(
                id=start_id + len(modules),
                node_ids=member_ids,
                component_id=component_id,
            ))
    return modules


def _cluster_component(
    component: SpectralComponent,
    start_id: int,
    n_modules: int | None,
) -> tuple[list[Module], int, float | None]:
    """Cluster one connected component and annotate module quality metrics."""
    X = component.eigenvectors
    if n_modules is None:
        n_modules = _estimate_k(X)
    n_modules = max(1, min(n_modules, len(component.node_ids)))

    if n_modules == 1:
        modules = [Module(
            id=start_id,
            node_ids=list(component.node_ids),
            component_id=component.id,
            cohesion=0.0,
            separation=None,
            confidence=0.5,
        )]
        return modules, 1, None

    labels = _kmeans(X, n_modules)
    silhouette = _silhouette_score(X, labels)
    modules = _labels_to_modules(
        labels,
        component.node_ids,
        n_modules,
        start_id=start_id,
        component_id=component.id,
    )
    _annotate_module_metrics(modules, X, labels, component.node_ids, silhouette)
    return modules, n_modules, silhouette


def _annotate_module_metrics(
    modules: list[Module],
    X: np.ndarray,
    labels: np.ndarray,
    node_ids: list[str],
    silhouette: float,
) -> None:
    """Compute cohesion, separation, and confidence for clustered modules."""
    centroids: dict[int, np.ndarray] = {}
    for module in modules:
        indices = [node_ids.index(node_id) for node_id in module.node_ids]
        centroids[module.id] = X[indices].mean(axis=0)

    for module in modules:
        indices = [node_ids.index(node_id) for node_id in module.node_ids]
        centroid = centroids[module.id]
        cohesion = float(np.mean(np.linalg.norm(X[indices] - centroid, axis=1)))

        other_distances = [
            float(np.linalg.norm(centroid - other_centroid))
            for other_id, other_centroid in centroids.items()
            if other_id != module.id
        ]
        separation = min(other_distances) if other_distances else None
        ratio = 0.5 if separation is None else separation / (separation + cohesion + 1e-9)
        confidence = max(0.0, min(1.0, 0.5 * max(silhouette, 0.0) + 0.5 * ratio))

        module.cohesion = cohesion
        module.separation = separation
        module.confidence = confidence


def _estimate_k(X: np.ndarray, max_k: int = 8) -> int:
    """Select k by the best silhouette score across candidate cluster counts."""
    max_k = min(max_k, X.shape[0] - 1)
    if max_k < 2:
        return 2

    scores = []
    for k in range(2, max_k + 1):
        labels = _kmeans(X, k)
        scores.append(_silhouette_score(X, labels))
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
        dist_sum = dists.sum()
        if dist_sum == 0:
            # All points are at existing centroids; fall back to uniform selection
            centroids.append(X[rng.integers(n)])
        else:
            probs = dists / dist_sum
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


def _is_degenerate(modules: list[Module], silhouette: float | None) -> bool:
    """Clustering is degenerate when it lacks meaningful structure.

    Two triggers, either sufficient:
    1. Clusters are too small (≤3 nodes on average) with weak silhouette.
       Three nodes per cluster lack statistical mass for spectral grouping
       to be more informative than package-based grouping.
    2. Silhouette is very poor (<0.3) regardless of cluster size.
       A silhouette below 0.3 indicates the clustering has no meaningful
       separation — essentially random assignment.

    In both cases, falling back to package grouping produces more stable
    and architecturally coherent modules.
    """
    if silhouette is not None and silhouette >= 0.5:
        return False
    clustered = [m for m in modules if not m.unassigned]
    if not clustered:
        return False
    total_nodes = sum(m.size for m in clustered)
    avg_size = total_nodes / len(clustered)
    if avg_size <= 3.0:
        return True
    if silhouette is not None and silhouette < 0.3:
        return True
    return False


def _package_grouping(node_ids: list[str]) -> list[Module]:
    """Group nodes by their top-level package (first dotted component).

    Confidence is set to 0.5 (medium): the grouping is certain (packages
    are well-defined), but we haven't validated structural cohesion
    through spectral analysis.
    """
    groups: dict[str, list[str]] = {}
    for nid in node_ids:
        pkg = nid.split(".", 1)[0]
        groups.setdefault(pkg, []).append(nid)
    return [
        Module(id=i, node_ids=sorted(members), confidence=0.5)
        for i, (_, members) in enumerate(sorted(groups.items()))
    ]
