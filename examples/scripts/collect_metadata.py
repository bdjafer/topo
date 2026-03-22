#!/usr/bin/env python3
"""
Collect metadata for a topo example from GitHub API + parsed artifacts.

Usage:
    python3 collect_metadata.py <name> <example_dir> <repo_url>

Writes metadata.json to <example_dir>/metadata.json.
"""

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def gh_api(endpoint: str) -> dict | None:
    """Call GitHub API via gh CLI. Returns None on failure."""
    try:
        result = subprocess.run(
            ["gh", "api", endpoint],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode == 0:
            return json.loads(result.stdout)
    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError):
        pass
    return None


def extract_owner_repo(url: str) -> str:
    """Extract 'owner/repo' from GitHub URL."""
    url = url.rstrip("/")
    if url.endswith(".git"):
        url = url[:-4]
    parts = url.split("/")
    return f"{parts[-2]}/{parts[-1]}"


def github_metadata(repo_url: str) -> dict:
    """Fetch metadata from GitHub API."""
    owner_repo = extract_owner_repo(repo_url)
    info = gh_api(f"repos/{owner_repo}")
    if not info:
        return {}

    return {
        "stars": info.get("stargazers_count", 0),
        "forks": info.get("forks_count", 0),
        "open_issues": info.get("open_issues_count", 0),
        "license": (info.get("license") or {}).get("spdx_id", "unknown"),
        "default_branch": info.get("default_branch", "main"),
        "created_at": info.get("created_at", ""),
        "updated_at": info.get("updated_at", ""),
        "topics": info.get("topics", []),
    }


def graph_metadata(example_dir: Path) -> dict:
    """Extract metadata from graph.json."""
    graph_path = example_dir / "graph.json"
    if not graph_path.exists():
        return {}

    with open(graph_path) as f:
        graph = json.load(f)

    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])

    node_kinds: dict[str, int] = {}
    for n in nodes:
        k = n.get("kind", "unknown")
        node_kinds[k] = node_kinds.get(k, 0) + 1

    edge_kinds: dict[str, int] = {}
    for e in edges:
        k = e.get("kind", "unknown")
        edge_kinds[k] = edge_kinds.get(k, 0) + 1

    files = {n.get("file", "") for n in nodes if n.get("file")}

    return {
        "nodes": len(nodes),
        "edges": len(edges),
        "node_kinds": node_kinds,
        "edge_kinds": edge_kinds,
        "files_parsed": len(files),
    }


def analysis_metadata(example_dir: Path) -> dict:
    """Extract metadata from analysis.json."""
    analysis_path = example_dir / "analysis.json"
    if not analysis_path.exists():
        return {}

    with open(analysis_path) as f:
        analysis = json.load(f)

    arch = analysis.get("architecture", {})
    modules = arch.get("modules", [])
    spectral = analysis.get("spectral") or {}
    health = analysis.get("health") or {}
    issues = analysis.get("issues", [])

    return {
        "spectral_modules": len(modules),
        "modularity_q": health.get("modularity_q"),
        "fiedler_value": spectral.get("fiedler_value"),
        "issue_count": len(issues),
        "components": spectral.get("components"),
        "semantic_enabled": analysis.get("semantic_enabled", False),
    }


def main() -> None:
    if len(sys.argv) < 4:
        print(f"Usage: {sys.argv[0]} <name> <example_dir> <repo_url>", file=sys.stderr)
        sys.exit(1)

    name = sys.argv[1]
    example_dir = Path(sys.argv[2])
    repo_url = sys.argv[3]

    metadata = {
        "name": name,
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "github": github_metadata(repo_url),
        "graph": graph_metadata(example_dir),
        "analysis": analysis_metadata(example_dir),
    }

    out_path = example_dir / "metadata.json"
    with open(out_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  Wrote {out_path}")


if __name__ == "__main__":
    main()
