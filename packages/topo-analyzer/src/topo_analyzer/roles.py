"""
Structural role classification.

Classifies code entities by their structural position in the graph:
hub, bridge, utility, entry point, orphan, etc.

Uses distribution-based classification: instead of hardcoded thresholds,
each node is classified by where it falls in the graph's own metric
distributions (percentile ranks of degree and betweenness, plus a
directional flow score).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum

import numpy as np

from topo_parser.graph import CodeGraph, EdgeKind


class StructuralRole(Enum):
    """Structural roles derived from graph position."""

    HUB = "hub"  # Highly connected — many dependents
    BRIDGE = "bridge"  # Connects otherwise separate regions
    UTILITY = "utility"  # Called by many, calls few — leaf-like service
    ENTRY_POINT = "entry_point"  # High in-degree from outside, low internal coupling
    ORPHAN = "orphan"  # Disconnected or nearly disconnected
    REGULAR = "regular"  # No distinctive structural role


@dataclass
class RoleAssignment:
    """A node's classified structural role with supporting metrics."""

    node_id: str
    role: StructuralRole
    degree: int
    betweenness: float
    in_degree: int
    out_degree: int


def classify_roles(
    graph: CodeGraph,
    edge_kind: EdgeKind = EdgeKind.CALLS,
    edge_kinds: list[EdgeKind] | None = None,
    *,
    pct_threshold: float = 0.9,
    betweenness_override: dict[str, float] | None = None,
) -> list[RoleAssignment]:
    """
    Classify structural roles for all nodes based on graph position.

    Uses distribution-based classification: roles are assigned based on
    where each node falls in the graph's own metric distributions rather
    than hardcoded absolute thresholds.

    Args:
        graph: The code graph.
        edge_kind: Which relationship layer to use (ignored if edge_kinds is set).
        edge_kinds: Multiple layers to combine for role classification.
        pct_threshold: Percentile rank above which a metric is considered
            unusually high (default 0.9 = top 10%).
        betweenness_override: Pre-computed betweenness centrality (e.g. from Rust).

    Returns:
        List of role assignments for each node.
    """
    kinds = edge_kinds if edge_kinds is not None else [edge_kind]

    # Build adjacency from all specified layers.
    node_ids = list(graph.nodes)
    node_set = set(node_ids)
    successors: dict[str, list[str]] = {nid: [] for nid in node_ids}
    predecessors: dict[str, list[str]] = {nid: [] for nid in node_ids}

    for ek in kinds:
        for edge in graph.edges_by_kind(ek):
            if edge.source in node_set and edge.target in node_set:
                successors[edge.source].append(edge.target)
                predecessors[edge.target].append(edge.source)

    if betweenness_override is not None:
        betweenness = betweenness_override
    else:
        betweenness = _brandes_betweenness(node_ids, successors, predecessors)

    # Collect per-node metrics.
    n = len(node_ids)
    degrees = np.zeros(n, dtype=int)
    in_degrees = np.zeros(n, dtype=int)
    out_degrees = np.zeros(n, dtype=int)
    btw_values = np.zeros(n)

    for i, node_id in enumerate(node_ids):
        in_degrees[i] = len(predecessors[node_id])
        out_degrees[i] = len(successors[node_id])
        degrees[i] = in_degrees[i] + out_degrees[i]
        btw_values[i] = betweenness.get(node_id, 0.0)

    # Compute population statistics.
    degree_pcts = _percentile_ranks(degrees)
    btw_pcts = _percentile_ranks(btw_values)
    median_degree = float(np.median(degrees))

    assignments = []
    for i, node_id in enumerate(node_ids):
        role = _classify_node(
            degree=int(degrees[i]),
            in_degree=int(in_degrees[i]),
            out_degree=int(out_degrees[i]),
            degree_pct=degree_pcts[i],
            betweenness_pct=btw_pcts[i],
            median_degree=median_degree,
            pct_threshold=pct_threshold,
        )
        assignments.append(RoleAssignment(
            node_id=node_id,
            role=role,
            degree=int(degrees[i]),
            betweenness=btw_values[i],
            in_degree=int(in_degrees[i]),
            out_degree=int(out_degrees[i]),
        ))

    return assignments


# Threshold above which we switch to approximate betweenness centrality.
# Exact betweenness is O(V*E); for large graphs we sample sqrt(n) pivots.
_BETWEENNESS_APPROX_THRESHOLD = 5000


def _brandes_betweenness(
    node_ids: list[str],
    successors: dict[str, list[str]],
    predecessors: dict[str, list[str]],
) -> dict[str, float]:
    """Brandes' algorithm for betweenness centrality on a directed graph.

    For large graphs (> _BETWEENNESS_APPROX_THRESHOLD nodes), samples
    sqrt(n) source nodes for approximation.
    """
    n = len(node_ids)
    if n == 0:
        return {}

    cb: dict[str, float] = {v: 0.0 for v in node_ids}

    # Choose source nodes: all for small graphs, sample for large.
    if n <= _BETWEENNESS_APPROX_THRESHOLD:
        sources = node_ids
    else:
        rng = np.random.RandomState(42)
        k = min(n - 1, max(2, int(n ** 0.5)))
        sources = list(rng.choice(node_ids, size=k, replace=False))

    for s in sources:
        # BFS from s.
        stack: list[str] = []
        pred: dict[str, list[str]] = {v: [] for v in node_ids}
        sigma: dict[str, int] = {v: 0 for v in node_ids}
        sigma[s] = 1
        dist: dict[str, int] = {v: -1 for v in node_ids}
        dist[s] = 0
        queue = deque([s])

        while queue:
            v = queue.popleft()
            stack.append(v)
            for w in successors[v]:
                if dist[w] < 0:
                    dist[w] = dist[v] + 1
                    queue.append(w)
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    pred[w].append(v)

        # Back-propagation of dependencies.
        delta: dict[str, float] = {v: 0.0 for v in node_ids}
        while stack:
            w = stack.pop()
            for v in pred[w]:
                delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w])
            if w != s:
                cb[w] += delta[w]

    # Normalize: for directed graphs, divide by (n-1)*(n-2).
    # If using sampling, scale up by n/k.
    if n > 2:
        scale = 1.0 / ((n - 1) * (n - 2))
        if n > _BETWEENNESS_APPROX_THRESHOLD:
            scale *= n / len(sources)
        for v in cb:
            cb[v] *= scale

    return cb


def _percentile_ranks(values: np.ndarray) -> np.ndarray:
    """Compute percentile rank for each value: fraction of values strictly less.

    Uses ranks/(n-1) so the maximum value maps to 1.0 and the minimum to 0.0.
    This ensures the top node is always classifiable regardless of graph size.

    Distribution-free, handles ties correctly, works with any shape
    including heavy-tailed and zero-inflated distributions.
    """
    n = len(values)
    if n <= 1:
        return np.zeros(n)
    sorted_vals = np.sort(values)
    # For each value, count how many are strictly less via searchsorted
    ranks = np.searchsorted(sorted_vals, values, side="left")
    return ranks / (n - 1)


# Structural constants that define what each role means.
# These are not statistical thresholds — they are definitional.
_MIN_DIRECTIONAL_DEGREE = 3  # Need ≥3 edges for direction to be meaningful
_DIRECTION_THRESHOLD = 0.6  # ≥60% flow imbalance = directional role
_MIN_HUB_GAP = 2  # Hub must exceed median by ≥2 (prevents false hubs in uniform graphs)


def _classify_node(
    degree: int,
    in_degree: int,
    out_degree: int,
    degree_pct: float,
    betweenness_pct: float,
    median_degree: float,
    pct_threshold: float,
) -> StructuralRole:
    """Classify a single node using distribution-based rules.

    Classification priority:
    1. ORPHAN:      degree == 0
    2. BRIDGE:      top-percentile betweenness but NOT top-percentile degree
    3. UTILITY:     strong sink (direction ≤ -0.6) with enough edges
    4. ENTRY_POINT: strong source (direction ≥ +0.6) with enough edges
    5. HUB:         top-percentile degree, above median gap, balanced flow
    6. REGULAR:     everything else
    """
    if degree == 0:
        return StructuralRole.ORPHAN

    direction = (out_degree - in_degree) / degree

    # Bridge: unusually high betweenness without unusually high degree
    if betweenness_pct >= pct_threshold and degree_pct < pct_threshold:
        return StructuralRole.BRIDGE

    # Directional roles: strong flow imbalance with enough edges
    if degree >= _MIN_DIRECTIONAL_DEGREE and direction <= -_DIRECTION_THRESHOLD:
        return StructuralRole.UTILITY

    if degree >= _MIN_DIRECTIONAL_DEGREE and direction >= _DIRECTION_THRESHOLD:
        return StructuralRole.ENTRY_POINT

    # Hub: top-percentile degree, clearly above median, balanced flow
    if (
        degree_pct >= pct_threshold
        and degree >= median_degree + _MIN_HUB_GAP
        and abs(direction) < _DIRECTION_THRESHOLD
    ):
        return StructuralRole.HUB

    return StructuralRole.REGULAR
