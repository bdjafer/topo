"""
Graph projection and scope selection for structural analysis.

The parser stores a rich, mixed-level graph. The analyzer works best when it
operates on an explicit projection of that graph: scoped to the code we care
about, restricted to the relationship layers we want, and lifted to a single
analysis level such as package, module, or symbol.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import tomllib

from topo_parser.graph import CodeGraph, Edge, EdgeKind, Node, NodeKind

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
    """The level at which the analyzer should reason about structure."""

    PACKAGE = "package"
    MODULE = "module"
    SYMBOL = "symbol"


@dataclass(frozen=True)
class AnalysisAnchor:
    """A file anchor that points back to the original graph node."""

    node_id: str
    file: str
    line: int
    kind: NodeKind

    def to_dict(self) -> dict[str, str | int]:
        """Serialize an anchor for JSON output."""
        return {
            "node_id": self.node_id,
            "file": self.file,
            "line": self.line,
            "kind": self.kind.value,
        }


@dataclass
class AnalysisPolicy:
    """Repo-level defaults for analysis execution."""

    path: Path
    scope: str | None = None
    level: AnalysisLevel | None = None
    ignores: dict[str, str] = field(default_factory=dict)


@dataclass
class AnalysisProjectionConfig:
    """Configuration for projecting a raw graph into an analysis graph."""

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
        """Build the default projection config for a single analysis run."""
        edge_kinds = DEFAULT_ANALYSIS_EDGE_KINDS if combined else (edge_kind,)
        layer_weights = DEFAULT_ANALYSIS_LAYER_WEIGHTS if combined else None
        return cls(
            level=level,
            edge_kinds=edge_kinds,
            layer_weights=layer_weights,
            scope_roots=scope_roots,
            internal_only=internal_only,
        )

    @property
    def scope_labels(self) -> list[str]:
        """Render scope roots as stable strings for summaries and JSON."""
        return [str(path) for path in self.scope_roots]


@dataclass
class AnalysisProjection:
    """A projected graph plus metadata that links back to the source graph."""

    graph: CodeGraph
    config: AnalysisProjectionConfig
    raw_node_count: int
    raw_edge_count: int
    scoped_node_count: int
    scoped_edge_count: int
    raw_to_projected: dict[str, str]
    node_anchors: dict[str, list[AnalysisAnchor]] = field(default_factory=dict)
    self_edge_count: int = 0

    def anchors_for(self, node_ids: list[str], limit: int = 3) -> list[AnalysisAnchor]:
        """Return representative anchors for one or more projected node IDs."""
        anchors: list[AnalysisAnchor] = []
        seen: set[tuple[str, int]] = set()
        for node_id in node_ids:
            for anchor in self.node_anchors.get(node_id, []):
                key = (str(anchor.file), anchor.line)
                if key in seen:
                    continue
                seen.add(key)
                anchors.append(anchor)
                if len(anchors) >= limit:
                    return anchors
        return anchors

    @property
    def scope_filtered_node_count(self) -> int:
        """How many parsed nodes were removed by scope filtering."""
        return max(0, self.raw_node_count - self.scoped_node_count)

    @property
    def scope_filtered_edge_count(self) -> int:
        """How many parsed edges were removed by scope filtering or layer selection."""
        return max(0, self.raw_edge_count - self.scoped_edge_count)

    @property
    def scope_node_ratio(self) -> float:
        """Fraction of parsed nodes that remain after scope filtering."""
        if self.raw_node_count == 0:
            return 0.0
        return self.scoped_node_count / self.raw_node_count

    @property
    def projection_node_ratio(self) -> float:
        """Compression ratio from scoped raw nodes to projected analysis nodes."""
        if self.scoped_node_count == 0:
            return 0.0
        return self.graph.node_count / self.scoped_node_count

    @property
    def self_edge_ratio(self) -> float:
        """Fraction of scoped edges that were dropped as self-edges."""
        if self.scoped_edge_count == 0:
            return 0.0
        return self.self_edge_count / self.scoped_edge_count


def discover_first_party_source_roots(
    root: Path,
    package_prefixes: tuple[str, ...] = DEFAULT_SCOPE_PREFIXES,
) -> tuple[Path, ...]:
    """Discover first-party `src/` roots from a uv-style workspace layout."""
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
                    "project",
                    {},
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


def build_projection(graph: CodeGraph, config: AnalysisProjectionConfig) -> AnalysisProjection:
    """Project a raw graph into a consistent analysis graph."""
    scope_roots = tuple(sorted(path.resolve() for path in config.scope_roots))
    selected_nodes = {
        node_id: node
        for node_id, node in graph.nodes.items()
        if node.kind in config.source_node_kinds
        and _is_in_scope(node.file, scope_roots)
    }
    module_nodes = {
        node_id
        for node_id, node in selected_nodes.items()
        if node.kind == NodeKind.MODULE
    }

    raw_to_projected: dict[str, str] = {}
    node_anchors: dict[str, list[AnalysisAnchor]] = {}
    projected_graph = CodeGraph()
    scoped_edge_count = 0
    self_edge_count = 0

    for node_id, node in selected_nodes.items():
        projected_id = _projected_node_id(node_id, node.kind, config.level, module_nodes)
        raw_to_projected[node_id] = projected_id
        anchor = AnalysisAnchor(
            node_id=node_id,
            file=node.file,
            line=node.line,
            kind=node.kind,
        )
        node_anchors.setdefault(projected_id, []).append(anchor)

    for projected_id, anchors in node_anchors.items():
        anchors.sort(key=lambda anchor: (str(anchor.file), anchor.line, anchor.node_id))
        representative = anchors[0]
        kind = representative.kind if config.level == AnalysisLevel.SYMBOL else NodeKind.MODULE
        projected_graph.add_node(Node(
            id=projected_id,
            kind=kind,
            file=representative.file,
            line=representative.line,
            name=projected_id.split(".")[-1],
        ))

    for edge in graph.edges:
        if edge.kind not in config.edge_kinds:
            continue
        if edge.source not in raw_to_projected or edge.target not in raw_to_projected:
            if config.internal_only:
                continue
            continue
        scoped_edge_count += 1
        source = raw_to_projected[edge.source]
        target = raw_to_projected[edge.target]
        if source == target:
            self_edge_count += 1
            continue
        projected_graph.add_edge(Edge(source=source, target=target, kind=edge.kind))

    return AnalysisProjection(
        graph=projected_graph,
        config=AnalysisProjectionConfig(
            level=config.level,
            edge_kinds=config.edge_kinds,
            layer_weights=dict(config.layer_weights) if config.layer_weights else None,
            scope_roots=scope_roots,
            internal_only=config.internal_only,
            source_node_kinds=config.source_node_kinds,
        ),
        raw_node_count=graph.node_count,
        raw_edge_count=graph.edge_count,
        scoped_node_count=len(selected_nodes),
        scoped_edge_count=scoped_edge_count,
        raw_to_projected=raw_to_projected,
        node_anchors=node_anchors,
        self_edge_count=self_edge_count,
    )


def _is_in_scope(file: str, scope_roots: tuple[Path, ...]) -> bool:
    """Check whether a source file is inside one of the configured scope roots."""
    if not scope_roots:
        return True
    resolved = Path(file).resolve()
    return any(resolved.is_relative_to(root) for root in scope_roots)


def find_analysis_policy_file(start_path: Path) -> Path | None:
    """Search upward for a repo-level analysis policy file."""
    current = start_path.resolve()
    if current.is_file():
        current = current.parent

    for directory in (current, *current.parents):
        for filename in POLICY_FILENAMES:
            candidate = directory / filename
            if candidate.is_file():
                return candidate
    return None


def load_analysis_policy(start_path: Path) -> AnalysisPolicy | None:
    """Load an analysis policy from the nearest repo-level TOML file."""
    policy_path = find_analysis_policy_file(start_path)
    if policy_path is None:
        return None

    try:
        payload = tomllib.loads(policy_path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"Invalid analysis policy in {policy_path}: {exc}") from exc

    analysis = payload.get("analysis", {})

    scope = analysis.get("scope")
    if scope is not None and scope not in {"auto", "all", "first-party"}:
        raise ValueError(f"Invalid analysis.scope in {policy_path}: {scope}")

    level_value = analysis.get("level")
    try:
        level = AnalysisLevel(level_value) if level_value is not None else None
    except ValueError as exc:
        raise ValueError(f"Invalid analysis.level in {policy_path}: {level_value}") from exc

    ignore_section = analysis.get("ignore", {})
    if not isinstance(ignore_section, dict):
        raise ValueError(f"analysis.ignore must be a table in {policy_path}")
    for key, value in ignore_section.items():
        if not isinstance(value, str):
            raise ValueError(
                f"analysis.ignore values must be strings (justification) in {policy_path}, "
                f"got {type(value).__name__} for key {key!r}"
            )

    return AnalysisPolicy(
        path=policy_path,
        scope=scope,
        level=level,
        ignores=dict(ignore_section),
    )


def _projected_node_id(
    node_id: str,
    node_kind: NodeKind,
    level: AnalysisLevel,
    module_nodes: set[str],
) -> str:
    """Map a raw node ID to the projected analysis node ID."""
    if level == AnalysisLevel.SYMBOL:
        return node_id

    module_id = _owning_module_id(node_id, node_kind, module_nodes)
    if level == AnalysisLevel.MODULE:
        return module_id
    return module_id.split(".", 1)[0]


def _owning_module_id(node_id: str, node_kind: NodeKind, module_nodes: set[str]) -> str:
    """Find the module node that owns a raw node."""
    if node_kind == NodeKind.MODULE:
        return node_id
    parts = node_id.split(".")
    for i in range(len(parts) - 1, 0, -1):
        candidate = ".".join(parts[:i])
        if candidate in module_nodes:
            return candidate
    return parts[0]
