"""Load and validate gold-standard architecture labels."""

from __future__ import annotations

import json
from pathlib import Path

from topo_parser.graph import CodeGraph


def load_gold_labels(codebase: str, labels_root: Path) -> dict:
    """Load gold labels for a codebase.

    Returns dict with:
        included_nodes: {node_id: arch_module}
        excluded_nodes: [node_id, ...]
        cross_directory_pairs: [(node_a, node_b), ...]  — pairs in same module, different dirs
    """
    label_file = labels_root / codebase / "labels.json"
    if not label_file.exists():
        raise FileNotFoundError(f"No gold labels for {codebase} at {label_file}")
    return json.loads(label_file.read_text())


def validate_gold_labels(
    labels: dict,
    graph: CodeGraph,
) -> dict:
    """Check how well gold labels cover the parsed graph.

    Returns validation report with coverage stats and warnings.
    """
    gold_nodes = set(labels.get("included_nodes", {}).keys())
    excluded = set(labels.get("excluded_nodes", []))
    graph_nodes = set(graph.nodes.keys())

    labeled = gold_nodes & graph_nodes
    unlabeled = graph_nodes - gold_nodes - excluded
    missing = gold_nodes - graph_nodes  # labels for nodes not in graph

    return {
        "graph_nodes": len(graph_nodes),
        "labeled_nodes": len(labeled),
        "unlabeled_nodes": len(unlabeled),
        "missing_from_graph": len(missing),
        "coverage": len(labeled) / len(graph_nodes) if graph_nodes else 0.0,
        "unlabeled_ids": sorted(unlabeled)[:20],  # First 20 for inspection
        "missing_ids": sorted(missing)[:20],
    }
