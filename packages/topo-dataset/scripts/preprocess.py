#!/usr/bin/env python3
"""
Batch preprocessing: export R-GIN features for all parsed repos.

For each repo with a graph.json, calls `topo export-features` to produce
features.npz + features.meta.json. Resumable, parallelizable, auditable.

Usage:
    python preprocess.py                          # All parsed repos
    python preprocess.py --repos flask,click      # Specific repos
    python preprocess.py --force                  # Re-process even if cached
    python preprocess.py --workers 4              # Parallel workers
    python preprocess.py --embeddings-dir /path   # Use pre-computed embeddings
"""

import argparse
import json
import subprocess
import sys
import tomllib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
EXAMPLES_DIR = PROJECT_ROOT / "examples"
REGISTRY = EXAMPLES_DIR / "registry.toml"
TOPO_BIN = PROJECT_ROOT / "target" / "release" / "topo"


def find_topo_binary() -> Path:
    """Find the topo binary (release or debug)."""
    if TOPO_BIN.exists():
        return TOPO_BIN
    debug = PROJECT_ROOT / "target" / "debug" / "topo"
    if debug.exists():
        return debug
    print("ERROR: topo binary not found. Run `cargo build -p topo-cli --release` first.", file=sys.stderr)
    sys.exit(1)


def load_registry() -> list[dict]:
    """Load repo entries from examples/registry.toml."""
    with open(REGISTRY, "rb") as f:
        reg = tomllib.load(f)
    return reg.get("example", [])


def registered_repos_with_graphs() -> list[str]:
    """Return names of registered repos that have graph.json parsed."""
    registry = load_registry()
    names = []
    for entry in registry:
        name = entry["name"]
        if (EXAMPLES_DIR / name / "graph.json").exists():
            names.append(name)
    return sorted(names)


def process_repo(
    name: str,
    topo_bin: str,
    resume: bool = True,
    embeddings_dir: str | None = None,
) -> dict:
    """Process one repo: run topo export-features. Returns status dict."""
    repo_dir = EXAMPLES_DIR / name
    graph_path = repo_dir / "graph.json"
    output_npz = repo_dir / "features.npz"
    output_meta = repo_dir / "features.meta.json"

    # Resume: skip if already processed
    if resume and output_npz.exists() and output_meta.exists():
        return {"repo": name, "status": "skipped"}

    if not graph_path.exists():
        return {"repo": name, "status": "not_parsed"}

    # Quick size check
    try:
        with open(graph_path) as f:
            graph = json.load(f)
        n_nodes = len(graph.get("nodes", []))
        n_edges = len(graph.get("edges", []))

        if n_nodes < 50:
            return {"repo": name, "status": "too_small", "n_nodes": n_nodes}
        if n_nodes > 50_000:
            return {"repo": name, "status": "too_large", "n_nodes": n_nodes}
    except (json.JSONDecodeError, OSError) as e:
        return {"repo": name, "status": "error", "error": f"Failed to read graph.json: {e}"}

    # Build topo export-features command
    cmd = [
        topo_bin,
        "export-features",
        "--input", str(graph_path),
        "-o", str(output_npz),
    ]

    # Optionally pass pre-computed embeddings
    emb_path = None
    if embeddings_dir:
        candidate = Path(embeddings_dir) / name / "embeddings.json"
        if candidate.exists():
            emb_path = candidate
    else:
        candidate = repo_dir / "embeddings.json"
        if candidate.exists():
            emb_path = candidate
    if emb_path:
        cmd.extend(["--embeddings", str(emb_path)])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            stderr = result.stderr.strip()
            return {"repo": name, "status": "error", "error": stderr[:500]}

        # Validate output exists
        if not output_npz.exists():
            return {"repo": name, "status": "error", "error": "NPZ not created"}
        if not output_meta.exists():
            return {"repo": name, "status": "error", "error": "meta.json not created"}

        return {
            "repo": name,
            "status": "ok",
            "n_nodes": n_nodes,
            "n_edges": n_edges,
        }

    except subprocess.TimeoutExpired:
        return {"repo": name, "status": "error", "error": "timeout (300s)"}
    except Exception as e:
        return {"repo": name, "status": "error", "error": str(e)}


def main():
    parser = argparse.ArgumentParser(
        description="Batch preprocessing: export R-GIN features for all parsed repos."
    )
    parser.add_argument(
        "--repos",
        type=str,
        default=None,
        help="Comma-separated repo names (default: all registered repos with graph.json)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-process even if features.npz exists",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1)",
    )
    parser.add_argument(
        "--embeddings-dir",
        type=str,
        default=None,
        help="Directory containing <name>/embeddings.json files",
    )
    args = parser.parse_args()

    topo_bin = str(find_topo_binary())
    resume = not args.force

    # Determine repo list from registry
    if args.repos:
        repo_names = [r.strip() for r in args.repos.split(",")]
    else:
        repo_names = registered_repos_with_graphs()

    if not repo_names:
        print("No parsed repos found. Run `make harvest` first.", file=sys.stderr)
        sys.exit(1)

    print(f"Processing {len(repo_names)} repos (workers={args.workers}, resume={resume})")

    results: list[dict] = []

    if args.workers <= 1:
        for name in repo_names:
            print(f"  [{name}]", end=" ", flush=True)
            try:
                r = process_repo(name, topo_bin, resume, args.embeddings_dir)
            except Exception as e:
                r = {"repo": name, "status": "error", "error": str(e)}
            print(r["status"])
            results.append(r)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(process_repo, name, topo_bin, resume, args.embeddings_dir): name
                for name in repo_names
            }
            for future in as_completed(futures):
                name = futures[future]
                try:
                    r = future.result()
                except Exception as e:
                    r = {"repo": name, "status": "error", "error": str(e)}
                print(f"  [{name}] {r['status']}")
                results.append(r)

    # Write quality report
    report_path = EXAMPLES_DIR / "quality_report.json"
    by_status: dict[str, int] = {}
    for r in results:
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1

    report = {
        "total": len(results),
        "summary": by_status,
        "repos": sorted(results, key=lambda r: r["repo"]),
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nResults: {by_status}")
    print(f"Quality report: {report_path}")

    errors = by_status.get("error", 0)
    if errors > 0:
        print(f"WARNING: {errors} repos failed", file=sys.stderr)


if __name__ == "__main__":
    main()
