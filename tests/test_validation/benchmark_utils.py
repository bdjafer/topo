"""Helpers for hermetic structural-analysis benchmarks."""

from __future__ import annotations

import json
import os
import tomllib
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from math import comb, log
from pathlib import Path

from topo_parser_python.graph import CodeGraph, EdgeKind, NodeKind
from topo_parser_python.python import parse_python_project

# ---------------------------------------------------------------------------
# Analysis configuration (formerly in topo_cli.projection)
# ---------------------------------------------------------------------------

DEFAULT_ANALYSIS_EDGE_KINDS: tuple[EdgeKind, ...] = (
    EdgeKind.CALLS,
    EdgeKind.IMPORTS,
    EdgeKind.INHERITS,
)
DEFAULT_ANALYSIS_LAYER_WEIGHTS: dict[EdgeKind, float] = {
    EdgeKind.CALLS: 1.0,
    EdgeKind.IMPORTS: 0.5,
    EdgeKind.INHERITS: 0.8,
}
DEFAULT_SCOPE_PREFIXES: tuple[str, ...] = ("topo-",)
POLICY_FILENAMES: tuple[str, ...] = ("topo.toml", ".topo.toml")


class AnalysisLevel(Enum):
    PACKAGE = "package"
    MODULE = "module"
    SYMBOL = "symbol"


@dataclass
class AnalysisPolicy:
    path: Path
    scope: str | None = None
    level: AnalysisLevel | None = None
    ignores: dict[str, str] = field(default_factory=dict)


@dataclass
class AnalysisProjectionConfig:
    level: AnalysisLevel = AnalysisLevel.SYMBOL
    edge_kinds: tuple[EdgeKind, ...] = (EdgeKind.CALLS,)
    layer_weights: dict[EdgeKind, float] | None = None
    scope_roots: tuple[Path, ...] = ()
    internal_only: bool = True
    source_node_kinds: tuple[NodeKind, ...] = (
        NodeKind.MODULE,
        NodeKind.CLASS,
        NodeKind.FUNCTION,
    )

    @classmethod
    def for_analysis(
        cls,
        *,
        edge_kind: EdgeKind,
        combined: bool,
        level: AnalysisLevel = AnalysisLevel.SYMBOL,
        scope_roots: tuple[Path, ...] = (),
        internal_only: bool = True,
    ) -> AnalysisProjectionConfig:
        edge_kinds = DEFAULT_ANALYSIS_EDGE_KINDS if combined else (edge_kind,)
        layer_weights = DEFAULT_ANALYSIS_LAYER_WEIGHTS if combined else None
        return cls(
            level=level,
            edge_kinds=edge_kinds,
            layer_weights=layer_weights,
            scope_roots=scope_roots,
            internal_only=internal_only,
        )


def discover_first_party_source_roots(
    root: Path,
    package_prefixes: tuple[str, ...] = DEFAULT_SCOPE_PREFIXES,
) -> tuple[Path, ...]:
    """Discover first-party ``src/`` roots from a uv-style workspace layout."""
    root = root.resolve()
    package_containers: list[Path] = []
    if (root / "packages").is_dir():
        package_containers.append(root / "packages")
    if root.name == "packages" and root.is_dir():
        package_containers.append(root)
    if not package_containers and (root / "pyproject.toml").exists() and (root / "src").is_dir():
        package_containers.append(root.parent)

    roots: list[Path] = []
    seen: set[Path] = set()
    for container in package_containers:
        for package_dir in sorted(path for path in container.iterdir() if path.is_dir()):
            pyproject = package_dir / "pyproject.toml"
            src_dir = package_dir / "src"
            if not pyproject.exists() or not src_dir.is_dir():
                continue
            try:
                project_name = tomllib.loads(pyproject.read_text(encoding="utf-8")).get(
                    "project", {},
                ).get("name", "")
            except tomllib.TOMLDecodeError:
                continue
            if not any(project_name.startswith(prefix) for prefix in package_prefixes):
                continue
            resolved = src_dir.resolve()
            if resolved not in seen:
                seen.add(resolved)
                roots.append(resolved)
    return tuple(sorted(roots))


def load_analysis_policy(start_path: Path) -> AnalysisPolicy | None:
    """Load an analysis policy from the nearest repo-level TOML file."""
    current = start_path.resolve()
    if current.is_file():
        current = current.parent

    policy_path = None
    for directory in (current, *current.parents):
        for filename in POLICY_FILENAMES:
            candidate = directory / filename
            if candidate.is_file():
                policy_path = candidate
                break
        if policy_path:
            break

    if policy_path is None:
        return None

    payload = tomllib.loads(policy_path.read_text(encoding="utf-8"))
    analysis = payload.get("analysis", {})
    scope = analysis.get("scope")
    level_value = analysis.get("level")
    level = AnalysisLevel(level_value) if level_value is not None else None
    ignore_section = analysis.get("ignore", {})

    return AnalysisPolicy(
        path=policy_path,
        scope=scope,
        level=level,
        ignores=dict(ignore_section) if isinstance(ignore_section, dict) else {},
    )


# ---------------------------------------------------------------------------
# Rust backend bridge (formerly in topo_cli._rust_backend)
# ---------------------------------------------------------------------------

try:
    import topo_analyzer
    _RUST_AVAILABLE = True
except ImportError:
    _RUST_AVAILABLE = False


def run_full_analysis(
    graph: CodeGraph,
    *,
    projection_config: AnalysisProjectionConfig,
    n_modules: int | None = None,
) -> dict:
    """Run the complete analysis pipeline via the Rust extension (PyO3)."""
    if not _RUST_AVAILABLE or not hasattr(topo_analyzer, "analyze_full"):
        raise RuntimeError("topo_analyzer.analyze_full is not available")

    nodes_json = [
        {"id": nid, "kind": node.kind.value, "file": node.file, "line": node.line}
        for nid, node in graph.nodes.items()
    ]
    edges_json = [
        {"source": edge.source, "target": edge.target, "kind": edge.kind.value}
        for edge in graph.edges
    ]

    weights_json = None
    if projection_config.layer_weights:
        weights_json = {k.value: w for k, w in projection_config.layer_weights.items() if w > 0}

    edge_kinds = [k.value for k in projection_config.edge_kinds]

    input_data: dict = {
        "nodes": nodes_json,
        "edges": edges_json,
        "edge_kinds": edge_kinds,
        "projection": {
            "level": projection_config.level.value,
            "source_node_kinds": [k.value for k in projection_config.source_node_kinds],
            "edge_kinds": edge_kinds,
            "scope_roots": [str(r) for r in projection_config.scope_roots],
            "internal_only": projection_config.internal_only,
        },
    }
    if n_modules is not None:
        input_data["k"] = n_modules
    if weights_json:
        input_data["layer_weights"] = weights_json

    result_json = topo_analyzer.analyze_full(json.dumps(input_data))
    return json.loads(result_json)


# ---------------------------------------------------------------------------
# Result wrappers
# ---------------------------------------------------------------------------

class StructuralAnalysisResult:
    """Thin wrapper around a Rust analysis dict to preserve attribute access."""

    def __init__(self, data: dict) -> None:
        self._data = data

    @property
    def modules(self) -> list[ModuleResult]:
        return [ModuleResult(m) for m in self._data.get("architecture", {}).get("modules", [])]

    @property
    def issues(self) -> list[dict]:
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


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

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
