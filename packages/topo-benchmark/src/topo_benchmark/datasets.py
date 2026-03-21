"""Discover and load benchmark datasets."""

from __future__ import annotations

import json
from pathlib import Path

from topo_parser_python.graph import CodeGraph
from topo_benchmark.codegraph_io import load_graph


def _default_dataset_root() -> Path:
    """Find the benchmark/datasets directory relative to the repo root."""
    # Walk up from this file to find the repo root
    current = Path(__file__).resolve()
    for parent in current.parents:
        candidate = parent / "benchmark" / "datasets"
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError("Cannot find benchmark/datasets directory")


def discover_cases(
    dimension: str,
    split: str | None = None,
    dataset_root: Path | None = None,
) -> list[Path]:
    """Discover all benchmark cases for a dimension, optionally filtered by split."""
    root = dataset_root or _default_dataset_root()
    dim_dir = root / dimension
    if not dim_dir.is_dir():
        return []

    cases = []
    for case_dir in sorted(dim_dir.iterdir()):
        if not case_dir.is_dir():
            continue
        meta_path = case_dir / "metadata.json"
        if not meta_path.exists():
            continue
        if split is not None:
            meta = json.loads(meta_path.read_text())
            if meta.get("split") != split:
                continue
        cases.append(case_dir)
    return cases


def load_metadata(case_dir: Path) -> dict:
    """Load metadata.json for a case."""
    return json.loads((case_dir / "metadata.json").read_text())


def load_architecture_case(case_dir: Path) -> tuple[CodeGraph, dict]:
    """Load an architecture recovery case: graph + gold labels."""
    graph = load_graph(case_dir / "graph.json")
    labels = json.loads((case_dir / "labels.json").read_text())
    return graph, labels


def load_mutation_case(case_dir: Path) -> tuple[dict[str, CodeGraph], dict]:
    """Load a mutation ranking case: variant graphs + expectations."""
    variants_dir = case_dir / "variants"
    variants: dict[str, CodeGraph] = {}
    for variant_path in sorted(variants_dir.glob("*.json")):
        name = variant_path.stem
        variants[name] = load_graph(variant_path)

    expectations = json.loads((case_dir / "expectations.json").read_text())
    return variants, expectations


def load_stability_case(case_dir: Path) -> tuple[CodeGraph, dict[str, CodeGraph], dict]:
    """Load a stability case: base graph + perturbations + node mappings."""
    base = load_graph(case_dir / "base_graph.json")
    perturbations: dict[str, CodeGraph] = {}
    pert_dir = case_dir / "perturbations"
    if pert_dir.is_dir():
        for pert_path in sorted(pert_dir.glob("*.json")):
            perturbations[pert_path.stem] = load_graph(pert_path)

    mapping = json.loads((case_dir / "node_mapping.json").read_text())
    return base, perturbations, mapping


def load_anomaly_case(case_dir: Path) -> tuple[CodeGraph, dict]:
    """Load an anomaly precision case: graph + gold annotations."""
    graph = load_graph(case_dir / "graph.json")
    gold = json.loads((case_dir / "gold.json").read_text())
    return graph, gold
