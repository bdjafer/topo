"""
Structural role classification.

Classifies code entities by their structural position in the graph:
hub, bridge, utility, entry point, orphan, etc.

Uses a combination of graph-theoretic metrics (degree, betweenness,
centrality) and spectral fingerprint properties.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import networkx as nx

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
) -> list[RoleAssignment]:
    """
    Classify structural roles for all nodes based on graph position.

    Uses only graph-local properties (degree, betweenness) — does not
    require spectral decomposition.  For large graphs, betweenness is
    approximated via random sampling to keep the cost manageable.

    Args:
        graph: The code graph.
        edge_kind: Which relationship layer to use (ignored if edge_kinds is set).
        edge_kinds: Multiple layers to combine for role classification.

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

    assignments = []
    for node_id in graph.nodes:
        in_deg = nx_graph.in_degree(node_id)
        out_deg = nx_graph.out_degree(node_id)
        deg = in_deg + out_deg
        btw = betweenness.get(node_id, 0.0)

        role = _classify_single(deg, in_deg, out_deg, btw, len(graph.nodes))
        assignments.append(RoleAssignment(
            node_id=node_id,
            role=role,
            degree=deg,
            betweenness=btw,
            in_degree=in_deg,
            out_degree=out_deg,
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


def _classify_single(
    degree: int,
    in_degree: int,
    out_degree: int,
    betweenness: float,
    n_nodes: int,
) -> StructuralRole:
    """Classify a single node based on its metrics.

    Thresholds are initial heuristics.  The hub threshold uses a log-scale
    cap so it remains meaningful on large graphs (the old linear 15%
    threshold would require degree > 7 000 on a 49K-node graph, making
    hub detection impossible in practice).
    """
    if degree == 0:
        return StructuralRole.ORPHAN

    # High betweenness + moderate degree = bridge
    if betweenness > 0.05 and degree < n_nodes * 0.3:
        return StructuralRole.BRIDGE

    # High degree = hub.
    # Use log-scale cap: for small graphs the threshold is ~15% of nodes,
    # for large graphs it plateaus so that well-connected nodes are still
    # detected as hubs.
    hub_threshold = max(5, min(n_nodes * 0.15, 10 * math.log2(max(n_nodes, 2))))
    if degree > hub_threshold:
        return StructuralRole.HUB

    # High in-degree, low out-degree = utility
    if in_degree > 3 and out_degree <= 1:
        return StructuralRole.UTILITY

    # High out-degree, low in-degree = entry point
    if out_degree > 3 and in_degree <= 1:
        return StructuralRole.ENTRY_POINT

    return StructuralRole.REGULAR
