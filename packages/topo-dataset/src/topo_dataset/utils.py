"""Shared utilities for topo-dataset."""

from pathlib import Path
import tomllib


def project_root() -> Path:
    """Return the topo project root (parent of packages/)."""
    here = Path(__file__).resolve()
    # src/topo_dataset/utils.py -> packages/topo-dataset -> packages -> root
    return here.parent.parent.parent.parent.parent


def examples_dir() -> Path:
    return project_root() / "examples"


def registry_path() -> Path:
    return examples_dir() / "registry.toml"


def load_registry() -> list[dict]:
    """Load all entries from examples/registry.toml."""
    path = registry_path()
    with open(path, "rb") as f:
        reg = tomllib.load(f)
    return reg.get("example", [])


def parsed_repos() -> list[str]:
    """Return names of repos that have graph.json."""
    edir = examples_dir()
    return sorted(
        d.name
        for d in edir.iterdir()
        if d.is_dir() and (d / "graph.json").exists()
    )
