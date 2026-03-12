"""
Structural role classification.

Classifies code entities by their structural position in the graph:
hub, bridge, utility, entry point, orphan, etc.

Uses a combination of graph-theoretic metrics (degree, betweenness,
centrality) and spectral fingerprint properties.
"""

from __future__ import annotations

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
) -> list[RoleAssignment]:
    """
    Classify structural roles for all nodes based on graph position.

    Args:
        graph: The code graph.
        edge_kind: Which relationship layer to use for role classification.

    Returns:
        List of role assignments for each node.
    """
    # Build directed NetworkX graph
    nx_graph = nx.DiGraph()
    for node_id in graph.nodes:
        nx_graph.add_node(node_id)

    for edge in graph.edges_by_kind(edge_kind):
        if edge.source in graph.nodes and edge.target in graph.nodes:
            nx_graph.add_edge(edge.source, edge.target)

    betweenness = nx.betweenness_centrality(nx_graph)

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


def _classify_single(
    degree: int,
    in_degree: int,
    out_degree: int,
    betweenness: float,
    n_nodes: int,
) -> StructuralRole:
    """Classify a single node based on its metrics. Thresholds are initial heuristics."""
    if degree == 0:
        return StructuralRole.ORPHAN

    # High betweenness + moderate degree = bridge
    if betweenness > 0.1 and degree < n_nodes * 0.3:
        return StructuralRole.BRIDGE

    # High degree = hub
    if degree > max(5, n_nodes * 0.15):
        return StructuralRole.HUB

    # High in-degree, low out-degree = utility
    if in_degree > 3 and out_degree <= 1:
        return StructuralRole.UTILITY

    # High out-degree, low in-degree = entry point
    if out_degree > 3 and in_degree <= 1:
        return StructuralRole.ENTRY_POINT

    return StructuralRole.REGULAR
