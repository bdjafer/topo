"""
Analysis configuration, policy loading, and scope discovery.

Provides the types and utilities the CLI needs to configure analysis runs:
analysis level, projection config, policy file loading, and monorepo scope
discovery. The actual projection (graph lifting) is done in Rust.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import tomllib

from topo_parser.graph import EdgeKind, NodeKind

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
