"""Experiment 2: Seeded Defect Detection.

Tests whether spectral analysis detects architectural violations that
directory grouping cannot. Compares detection rates across methods.

STATUS: Scaffold — mutation operators and scoring logic need implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from topo_parser.graph import CodeGraph, Edge, EdgeKind


# --- Mutation operators ---


def mutate_reverse_dependency(
    graph: CodeGraph,
    source_mod: str,
    target_mod: str,
) -> CodeGraph:
    """Add a call edge from a low-level module to a high-level module.

    Violates layering: the dependency should only flow one direction.
    """
    raise NotImplementedError("Experiment 2 not yet implemented")


def mutate_add_cycle(
    graph: CodeGraph,
    mod_a: str,
    mod_b: str,
) -> CodeGraph:
    """Add mutual import edges between two previously-independent modules."""
    raise NotImplementedError("Experiment 2 not yet implemented")


def mutate_god_object(
    graph: CodeGraph,
    target_mod: str,
    source_mods: list[str],
    n_functions: int = 5,
) -> CodeGraph:
    """Move functions from multiple modules into one module."""
    raise NotImplementedError("Experiment 2 not yet implemented")


def mutate_misplace_utility(
    graph: CodeGraph,
    node_id: str,
    target_mod: str,
) -> CodeGraph:
    """Move a utility function to an unrelated module."""
    raise NotImplementedError("Experiment 2 not yet implemented")


def mutate_boundary_erosion(
    graph: CodeGraph,
    mod_a: str,
    mod_b: str,
    n_edges: int = 3,
) -> CodeGraph:
    """Add sparse cross-module call edges that shouldn't exist."""
    raise NotImplementedError("Experiment 2 not yet implemented")


# --- Detection scoring ---


@dataclass
class MutationDetectionResult:
    mutation_type: str
    codebase: str
    spectral_detected: bool = False
    directory_detected: bool = False
    louvain_detected: bool = False
    spectral_false_positives: int = 0
    directory_false_positives: int = 0
    louvain_false_positives: int = 0


@dataclass
class Experiment2Result:
    per_mutation: list[MutationDetectionResult] = field(default_factory=list)
    verdict: str = "NOT_IMPLEMENTED"
    verdict_details: list[str] = field(default_factory=list)


def run_experiment_2(
    codebases_root: Path,
    labels_root: Path,
    output_dir: Path,
) -> Experiment2Result:
    """Run Experiment 2: Seeded Defect Detection."""
    return Experiment2Result(
        verdict="NOT_IMPLEMENTED",
        verdict_details=["Experiment 2 scaffold — mutation operators need implementation."],
    )
