"""Experiment 4: Stability Under Architecture-Preserving Transformations.

Tests whether spectral output is robust (stable under irrelevant changes,
sensitive to real changes). If not, the signal is noise.

STATUS: Scaffold — graph transformations and comparison logic need implementation.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path

from topo_parser.graph import CodeGraph


# --- Architecture-preserving transformations ---


def rename_nodes(graph: CodeGraph) -> tuple[CodeGraph, dict[str, str]]:
    """Rename all nodes to generic IDs (f1, f2, ...) preserving graph structure.

    Returns (transformed_graph, old_to_new_mapping).
    """
    raise NotImplementedError("Experiment 4 not yet implemented")


def reorder_edges(graph: CodeGraph, seed: int = 42) -> CodeGraph:
    """Shuffle edge ordering. Should not affect spectral output."""
    raise NotImplementedError("Experiment 4 not yet implemented")


# --- Architecture-breaking transformations ---


def move_node(
    graph: CodeGraph,
    node_id: str,
    new_module: str,
) -> CodeGraph:
    """Move a node from its current module to a different one.

    Rewires all edges to/from the node to use the new module ID.
    """
    raise NotImplementedError("Experiment 4 not yet implemented")


def merge_modules(
    graph: CodeGraph,
    mod_a: str,
    mod_b: str,
) -> CodeGraph:
    """Merge two modules into one by renaming all mod_b nodes to mod_a."""
    raise NotImplementedError("Experiment 4 not yet implemented")


def add_cycle_edges(
    graph: CodeGraph,
    mod_a: str,
    mod_b: str,
) -> CodeGraph:
    """Add bidirectional dependency edges between two modules."""
    raise NotImplementedError("Experiment 4 not yet implemented")


# --- Comparison ---


@dataclass
class TransformationResult:
    name: str
    type: str  # "preserving" or "breaking"
    partition_ari: float = 0.0
    role_f1: float = 0.0
    anomaly_jaccard: float = 0.0
    error: str | None = None


@dataclass
class Experiment4Result:
    transformations: list[TransformationResult] = field(default_factory=list)
    mean_preserving_ari: float = 0.0
    mean_breaking_ari: float = 0.0
    separation: float = 0.0
    verdict: str = "NOT_IMPLEMENTED"
    verdict_details: list[str] = field(default_factory=list)


def run_experiment_4(
    codebases_root: Path,
    labels_root: Path,
    output_dir: Path,
) -> Experiment4Result:
    """Run Experiment 4: Stability Under Transformations."""
    return Experiment4Result(
        verdict="NOT_IMPLEMENTED",
        verdict_details=["Experiment 4 scaffold — transformations need implementation."],
    )
