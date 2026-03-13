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

from dataclasses import dataclass
from enum import Enum

import networkx as nx
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

    Returns:
        List of role assignments for each node.
    """
    kinds = edge_kinds if edge_kinds is not None else [edge_kind]

    # Build directed NetworkX graph from all specified layers
    nx_graph = nx.DiGraph()
    for node_id in graph.nodes:
        nx_graph.add_node(node_id)

    for ek in kinds:
        for edge in graph.edges_by_kind(ek):
            if edge.source in graph.nodes and edge.target in graph.nodes:
                nx_graph.add_edge(edge.source, edge.target)

    betweenness = _compute_betweenness(nx_graph)

    # Collect per-node metrics
    node_ids = list(graph.nodes)
    n = len(node_ids)
    degrees = np.zeros(n, dtype=int)
    in_degrees = np.zeros(n, dtype=int)
    out_degrees = np.zeros(n, dtype=int)
    btw_values = np.zeros(n)

    for i, node_id in enumerate(node_ids):
        in_degrees[i] = nx_graph.in_degree(node_id)
        out_degrees[i] = nx_graph.out_degree(node_id)
        degrees[i] = in_degrees[i] + out_degrees[i]
        btw_values[i] = betweenness.get(node_id, 0.0)

    # Compute population statistics
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
# Exact betweenness is O(V*E); for a 49K-node / 375K-edge graph that is
# ~18 billion operations — too slow in practice.
_BETWEENNESS_APPROX_THRESHOLD = 5000


def _compute_betweenness(nx_graph: nx.DiGraph) -> dict[str, float]:
    """Compute betweenness centrality, using approximation for large graphs."""
    n = nx_graph.number_of_nodes()
    if n <= _BETWEENNESS_APPROX_THRESHOLD:
        return nx.betweenness_centrality(nx_graph)
    # Sample sqrt(n) pivot nodes — gives a good accuracy/speed trade-off
    # while keeping cost at O(k * E) where k = sqrt(n).
    k = min(n - 1, max(2, int(n ** 0.5)))
    return nx.betweenness_centrality(nx_graph, k=k)


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
