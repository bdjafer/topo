"""Helpers for hermetic structural-analysis benchmarks."""

from __future__ import annotations

import os
from collections import defaultdict
from math import comb, log
from pathlib import Path

from topo_cli._rust_backend import run_full_analysis
from topo_cli.projection import (
    AnalysisLevel,
    AnalysisProjectionConfig,
    discover_first_party_source_roots,
    load_analysis_policy,
)
from topo_parser.graph import CodeGraph, EdgeKind
from topo_parser.python import parse_python_project


class StructuralAnalysisResult:
    """Thin wrapper around a Rust analysis dict to preserve attribute access."""

    def __init__(self, data: dict) -> None:
        self._data = data

    @property
    def modules(self) -> list[ModuleResult]:
        return [ModuleResult(m) for m in self._data.get("architecture", {}).get("modules", [])]

    @property
    def findings(self) -> list[dict]:
        return self._data.get("issues", [])

    @property
    def anomalies(self) -> list[dict]:
        return self._data.get("anomalies", [])

    @property
    def roles(self) -> list[dict]:
        return self._data.get("roles", [])

    @property
    def health(self) -> dict:
        return self._data.get("health", {})

    @property
    def coverage(self) -> dict:
        return self._data.get("coverage", {})

    @property
    def cross_package_dependencies(self) -> list[dict]:
        return self._data.get("architecture", {}).get("dependencies", [])

    @property
    def raw(self) -> dict:
        return self._data


class ModuleResult:
    """Thin wrapper around a module dict."""

    def __init__(self, data: dict) -> None:
        self._data = data

    @property
    def id(self) -> int:
        return self._data["id"]

    @property
    def label(self) -> str:
        return self._data.get("label", "")

    @property
    def node_ids(self) -> list[str]:
        return self._data.get("members", [])

    @property
    def size(self) -> int:
        return self._data.get("size", 0)

    @property
    def unassigned(self) -> bool:
        return self._data.get("unassigned", False)

    @property
    def cohesion(self) -> float | None:
        return self._data.get("cohesion")

    @property
    def separation(self) -> float | None:
        return self._data.get("separation")

    @property
    def confidence(self) -> float | None:
        return self._data.get("confidence")


def repo_root() -> Path:
    """Return the repository root for self-analysis tests."""
    return Path(__file__).resolve().parents[2]


def fixture_root(name: str) -> Path:
    """Return the root directory of a validation fixture."""
    return repo_root() / "tests" / "fixtures" / "validation" / name


def _ensure_rust_backend() -> None:
    """Set the Rust backend environment variable."""
    os.environ["TOPO_BACKEND"] = "rust"


def analyze_fixture(
    name: str,
    *,
    level: AnalysisLevel = AnalysisLevel.MODULE,
    combined: bool = True,
    n_modules: int | None = None,
) -> StructuralAnalysisResult:
    """Parse and analyze a committed benchmark fixture via Rust."""
    _ensure_rust_backend()
    root = fixture_root(name)
    graph = parse_python_project(root)
    config = AnalysisProjectionConfig.for_analysis(
        edge_kind=EdgeKind.CALLS,
        combined=combined,
        level=level,
    )
    data = run_full_analysis(graph, projection_config=config, n_modules=n_modules)
    return StructuralAnalysisResult(data)


def analyze_graph(
    graph: CodeGraph,
    *,
    level: AnalysisLevel = AnalysisLevel.MODULE,
    combined: bool = True,
    n_modules: int | None = None,
) -> StructuralAnalysisResult:
    """Analyze an in-memory graph using the benchmark defaults via Rust."""
    _ensure_rust_backend()
    config = AnalysisProjectionConfig.for_analysis(
        edge_kind=EdgeKind.CALLS,
        combined=combined,
        level=level,
    )
    data = run_full_analysis(graph, projection_config=config, n_modules=n_modules)
    return StructuralAnalysisResult(data)


def analyze_topo_self() -> StructuralAnalysisResult:
    """Run topo on its own first-party source roots via Rust."""
    _ensure_rust_backend()
    root = repo_root()
    policy = load_analysis_policy(root)
    scope_setting = policy.scope if policy and policy.scope else "auto"
    if scope_setting == "all":
        scope_roots = ()
    else:
        scope_roots = discover_first_party_source_roots(root)
    graph = parse_python_project(root, include_roots=list(scope_roots))
    config = AnalysisProjectionConfig.for_analysis(
        edge_kind=EdgeKind.CALLS,
        combined=True,
        level=policy.level if policy and policy.level else AnalysisLevel.MODULE,
        scope_roots=scope_roots,
    )
    data = run_full_analysis(graph, projection_config=config)
    return StructuralAnalysisResult(data)


def labels_from_modules(result: StructuralAnalysisResult) -> dict[str, int]:
    """Extract module labels for all assigned nodes in an analysis result."""
    labels: dict[str, int] = {}
    for module in result.modules:
        if module.unassigned:
            continue
        for node_id in module.node_ids:
            labels[node_id] = module.id
    return labels


def labels_by_top_package(node_ids: list[str]) -> dict[str, str]:
    """Use the top-level package as a simple architectural baseline."""
    return {node_id: node_id.split(".", 1)[0] for node_id in node_ids}


def compute_nmi_from_mappings(
    left_labels: dict[str, object],
    right_labels: dict[str, object],
) -> float:
    """Compute normalized mutual information between two label mappings."""
    common = sorted(set(left_labels) & set(right_labels))
    if not common:
        return 0.0
    return _compute_nmi(
        [left_labels[node_id] for node_id in common],
        [right_labels[node_id] for node_id in common],
    )


def compute_ari_from_mappings(
    left_labels: dict[str, object],
    right_labels: dict[str, object],
) -> float:
    """Compute adjusted Rand index between two label mappings."""
    common = sorted(set(left_labels) & set(right_labels))
    if not common:
        return 0.0
    return _compute_ari(
        [left_labels[node_id] for node_id in common],
        [right_labels[node_id] for node_id in common],
    )


def _compute_nmi(labels_a: list[object], labels_b: list[object]) -> float:
    """Normalized Mutual Information between two clusterings."""
    n = len(labels_a)
    assert n == len(labels_b) and n > 0

    joint: dict[tuple[object, object], int] = defaultdict(int)
    count_a: dict[object, int] = defaultdict(int)
    count_b: dict[object, int] = defaultdict(int)
    for label_a, label_b in zip(labels_a, labels_b):
        joint[(label_a, label_b)] += 1
        count_a[label_a] += 1
        count_b[label_b] += 1

    mutual_information = 0.0
    for (label_a, label_b), nij in joint.items():
        if nij == 0:
            continue
        pij = nij / n
        pi = count_a[label_a] / n
        pj = count_b[label_b] / n
        mutual_information += pij * log(pij / (pi * pj))

    entropy_a = -sum((count / n) * log(count / n) for count in count_a.values() if count > 0)
    entropy_b = -sum((count / n) * log(count / n) for count in count_b.values() if count > 0)

    if entropy_a + entropy_b == 0:
        return 1.0
    return 2 * mutual_information / (entropy_a + entropy_b)


def _compute_ari(labels_a: list[object], labels_b: list[object]) -> float:
    """Adjusted Rand Index between two clusterings."""
    n = len(labels_a)
    assert n == len(labels_b) and n > 0
    if n < 2:
        return 1.0

    joint: dict[tuple[object, object], int] = defaultdict(int)
    count_a: dict[object, int] = defaultdict(int)
    count_b: dict[object, int] = defaultdict(int)
    for label_a, label_b in zip(labels_a, labels_b):
        joint[(label_a, label_b)] += 1
        count_a[label_a] += 1
        count_b[label_b] += 1

    total_pairs = comb(n, 2)
    sum_joint = sum(comb(count, 2) for count in joint.values() if count >= 2)
    sum_a = sum(comb(count, 2) for count in count_a.values() if count >= 2)
    sum_b = sum(comb(count, 2) for count in count_b.values() if count >= 2)

    expected_index = (sum_a * sum_b) / total_pairs if total_pairs else 0.0
    max_index = 0.5 * (sum_a + sum_b)
    denominator = max_index - expected_index
    if denominator == 0:
        return 1.0
    return (sum_joint - expected_index) / denominator
